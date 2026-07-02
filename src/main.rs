use clap::Parser;
use img_hash::image;
use img_hash::image::GenericImageView;
use img_hash::{HashAlg, HasherConfig, ImageHash};
#[cfg(not(windows))]
use libheif_rs::{ColorSpace, HeifContext, LibHeif, RgbChroma};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Stand-in for `libheif_rs::LibHeif` on Windows, where HEIC decoding isn't wired up
/// (libheif has no prebuilt Windows binaries available in this build environment).
#[cfg(windows)]
struct LibHeif;

#[cfg(windows)]
impl LibHeif {
    fn new() -> Self {
        LibHeif
    }
}

/// Find visually similar / near-duplicate photos in a directory using perceptual hashing,
/// and generate an HTML report with thumbnails for manual review.
#[derive(Parser, Debug)]
#[command(name = "simphoto", version, about)]
struct Args {
    /// Directory to scan for images
    dir: PathBuf,

    /// Maximum Hamming distance between hashes (0-256) to consider two images similar
    /// (0 = exact hash match)
    #[arg(short, long, default_value_t = 40)]
    threshold: u32,

    /// Do not recurse into subdirectories
    #[arg(long)]
    no_recursive: bool,

    /// Path to write the HTML report (default: <dir>/similar_photos_report.html)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Thumbnail size in pixels (longest side)
    #[arg(long, default_value_t = 220)]
    thumb_size: u32,
}

const EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff", "tif", "heic", "heif",
];

struct Entry {
    path: PathBuf,
    size: u64,
    width: u32,
    height: u32,
    hash: ImageHash,
}

fn ext_lower(path: &Path) -> Option<String> {
    path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase())
}

fn is_image(path: &Path) -> bool {
    ext_lower(path)
        .map(|e| EXTENSIONS.contains(&e.as_str()))
        .unwrap_or(false)
}

fn is_heic(path: &Path) -> bool {
    ext_lower(path)
        .map(|e| e == "heic" || e == "heif")
        .unwrap_or(false)
}

/// Why decoding an image file failed, so callers can report the real cause instead of
/// re-deriving it (e.g. re-checking the platform and extension to guess "must be HEIC").
enum OpenError {
    Io,
    UnsupportedHeic,
    Decode,
}

/// Decode a HEIC/HEIF file into a `DynamicImage` via libheif, since the `image` crate
/// (pinned to 0.23 for compatibility with `img_hash`) has no HEIC decoder.
#[cfg(not(windows))]
fn open_heic(lib_heif: &LibHeif, bytes: &[u8]) -> Result<image::DynamicImage, OpenError> {
    let ctx = HeifContext::read_from_bytes(bytes).map_err(|_| OpenError::Decode)?;
    let handle = ctx.primary_image_handle().map_err(|_| OpenError::Decode)?;
    let heif_img = lib_heif
        .decode(&handle, ColorSpace::Rgb(RgbChroma::Rgb), None)
        .map_err(|_| OpenError::Decode)?;
    let width = heif_img.width();
    let height = heif_img.height();
    let planes = heif_img.planes();
    let plane = planes.interleaved.ok_or(OpenError::Decode)?;
    let row_bytes = (width * 3) as usize;
    let mut buf = vec![0u8; row_bytes * height as usize];
    for y in 0..height as usize {
        let src = &plane.data[y * plane.stride..y * plane.stride + row_bytes];
        buf[y * row_bytes..(y + 1) * row_bytes].copy_from_slice(src);
    }
    let rgb = image::RgbImage::from_raw(width, height, buf).ok_or(OpenError::Decode)?;
    Ok(image::DynamicImage::ImageRgb8(rgb))
}

#[cfg(windows)]
fn open_heic(_lib_heif: &LibHeif, _bytes: &[u8]) -> Result<image::DynamicImage, OpenError> {
    Err(OpenError::UnsupportedHeic)
}

/// Read the EXIF `Orientation` tag (1-8, default 1 = upright) from already-loaded file
/// bytes. `kamadak-exif` natively understands the JPEG/TIFF/HEIF/PNG/WebP containers, so
/// this works for every format this tool handles, including HEIC photos from phones.
fn read_exif_orientation(bytes: &[u8]) -> u32 {
    let mut cursor = std::io::Cursor::new(bytes);
    let exif = match exif::Reader::new().read_from_container(&mut cursor) {
        Ok(e) => e,
        Err(_) => return 1,
    };
    exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        .and_then(|f| f.value.get_uint(0))
        .unwrap_or(1)
}

/// Apply the pixel transform implied by an EXIF orientation value so that images shot
/// in different physical orientations (e.g. portrait vs. landscape) hash consistently
/// instead of hashing as if they were unrelated pictures.
fn apply_exif_orientation(img: image::DynamicImage, orientation: u32) -> image::DynamicImage {
    match orientation {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.fliph().rotate270(),
        6 => img.rotate90(),
        7 => img.fliph().rotate90(),
        8 => img.rotate270(),
        _ => img,
    }
}

/// Read a file once and decode both its pixels and its EXIF orientation from that single
/// buffer, instead of letting the image decoder and the EXIF reader each open and read
/// the file independently.
fn open_image(lib_heif: &LibHeif, path: &Path) -> Result<image::DynamicImage, OpenError> {
    let bytes = std::fs::read(path).map_err(|_| OpenError::Io)?;
    let img = if is_heic(path) {
        open_heic(lib_heif, &bytes)?
    } else {
        image::load_from_memory(&bytes).map_err(|_| OpenError::Decode)?
    };
    Ok(apply_exif_orientation(img, read_exif_orientation(&bytes)))
}

fn collect_paths(dir: &Path, recursive: bool) -> Vec<PathBuf> {
    let walker = WalkDir::new(dir).max_depth(if recursive { usize::MAX } else { 1 });
    walker
        .into_iter()
        .filter_map(|e| e.ok())
        .map(|e| e.into_path())
        .filter(|p| p.is_file() && is_image(p))
        .collect()
}

fn format_bytes(n: u64) -> String {
    let n = n as f64;
    if n >= 1e9 {
        format!("{:.2} GB", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.2} MB", n / 1e6)
    } else if n >= 1e3 {
        format!("{:.1} KB", n / 1e3)
    } else {
        format!("{} B", n as u64)
    }
}

/// Bytes reclaimed if every member of `group` is deleted except the largest (the
/// suggested keeper). Assumes `group` is sorted by size descending.
fn group_reclaim(entries: &[Entry], group: &[usize]) -> u64 {
    let total: u64 = group.iter().map(|&i| entries[i].size).sum();
    total - entries[group[0]].size
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Percent-encode a filesystem path into a `file://` URL so it can be opened by clicking
/// a link in the report (handles spaces, non-ASCII, etc.).
fn file_url(path: &Path) -> String {
    let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let s = abs.to_string_lossy();
    let mut out = String::from("file://");
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

fn generate_thumbnail(
    lib_heif: &LibHeif,
    path: &Path,
    thumb_dir: &Path,
    idx: usize,
    size: u32,
) -> Option<String> {
    let img = open_image(lib_heif, path).ok()?;
    let file_name = format!("thumb_{}.jpg", idx);
    let out_path = thumb_dir.join(&file_name);
    img.thumbnail(size, size)
        .to_rgb8()
        .save(&out_path)
        .ok()?;
    Some(file_name)
}

fn main() {
    let args = Args::parse();

    if !args.dir.is_dir() {
        eprintln!("Error: {:?} is not a directory", args.dir);
        std::process::exit(1);
    }

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| args.dir.join("similar_photos_report.html"));
    let thumb_dir_name = format!(
        "{}_thumbs",
        output.file_stem().unwrap_or_default().to_string_lossy()
    );
    let thumb_dir = output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&thumb_dir_name);

    let paths = collect_paths(&args.dir, !args.no_recursive);
    if paths.is_empty() {
        println!("No image files found in {:?}", args.dir);
        return;
    }
    println!("Hashing {} images...", paths.len());

    let lib_heif = LibHeif::new();

    let entries: Vec<Entry> = paths
        .par_iter()
        .filter_map(|path| {
            let img = match open_image(&lib_heif, path) {
                Ok(img) => img,
                Err(OpenError::UnsupportedHeic) => {
                    eprintln!(
                        "Skipping {:?}: HEIC/HEIF is not supported in this build",
                        path
                    );
                    return None;
                }
                Err(_) => {
                    eprintln!("Skipping {:?}: failed to decode", path);
                    return None;
                }
            };
            let (width, height) = (img.width(), img.height());
            let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            // DCT preprocessing (classic "pHash") was tried here but it focuses on the
            // low-frequency "broad strokes" of an image, which caused unrelated real
            // photos with similar lighting/composition to hash too close together.
            // Plain Gradient (dHash) compares local pixel-to-pixel transitions instead,
            // which is more discriminative for real photo content; the larger 16x16
            // hash (vs. the 8x8 default) still gives finer-grained distances than before.
            let hasher = HasherConfig::new()
                .hash_alg(HashAlg::Gradient)
                .hash_size(16, 16)
                .to_hasher();
            let hash = hasher.hash_image(&img);
            Some(Entry {
                path: path.clone(),
                size,
                width,
                height,
                hash,
            })
        })
        .collect();

    println!(
        "Comparing {} hashes (threshold <= {})...",
        entries.len(),
        args.threshold
    );

    // Cluster around representatives instead of taking the transitive closure of
    // "similar" pairs: with a plain union-find, A~B and B~C being similar would merge
    // A and C into one group even when A and C are nothing alike (chaining through a
    // "bridge" photo). Processing largest-first and only matching directly against an
    // unclaimed representative keeps every group member genuinely close to it.
    let n = entries.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(entries[i].size));

    let mut assigned = vec![false; n];
    let mut duplicate_groups: Vec<Vec<usize>> = Vec::new();
    for &i in &order {
        if assigned[i] {
            continue;
        }
        assigned[i] = true;
        // A plain sequential scan beats `par_iter` here: a hash `dist` is a handful of
        // XOR+popcount ops, so rayon's per-call task-splitting overhead would dominate
        // the actual work for most representatives.
        let matches: Vec<usize> = order
            .iter()
            .cloned()
            .filter(|&j| j != i && !assigned[j] && entries[i].hash.dist(&entries[j].hash) <= args.threshold)
            .collect();
        if matches.is_empty() {
            continue;
        }
        let mut group = vec![i];
        for j in matches {
            assigned[j] = true;
            group.push(j);
        }
        duplicate_groups.push(group);
    }

    // Sort members within each group by size descending (largest = suggested keeper),
    // then sort groups by reclaimable space descending (biggest cleanup wins first).
    for group in &mut duplicate_groups {
        group.sort_by_key(|&i| std::cmp::Reverse(entries[i].size));
    }
    duplicate_groups.sort_by_key(|g| std::cmp::Reverse(group_reclaim(&entries, g)));

    if duplicate_groups.is_empty() {
        println!("No similar photo groups found.");
        return;
    }

    let total_dup_files: usize = duplicate_groups.iter().map(|g| g.len()).sum();
    let reclaimable: u64 = duplicate_groups
        .iter()
        .map(|g| group_reclaim(&entries, g))
        .sum();

    println!(
        "\nFound {} group(s), {} duplicate file(s), ~{} reclaimable.\n",
        duplicate_groups.len(),
        total_dup_files,
        format_bytes(reclaimable)
    );

    println!("Generating thumbnails...");
    std::fs::create_dir_all(&thumb_dir).expect("failed to create thumbnail directory");
    let thumb_names: HashMap<usize, String> = duplicate_groups
        .iter()
        .flatten()
        .cloned()
        .collect::<Vec<usize>>()
        .par_iter()
        .filter_map(|&idx| {
            generate_thumbnail(&lib_heif, &entries[idx].path, &thumb_dir, idx, args.thumb_size)
                .map(|name| (idx, name))
        })
        .collect();

    let mut html = String::new();
    html.push_str("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">");
    html.push_str("<title>Similar Photos Report</title><style>");
    html.push_str(
        "body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:24px}\
         h1{margin-top:0}\
         .summary{background:#1c1c1c;padding:16px;border-radius:8px;margin-bottom:24px}\
         .group{background:#1c1c1c;border-radius:8px;padding:16px;margin-bottom:20px}\
         .group h2{margin:0 0 4px 0;font-size:16px}\
         .group .meta{color:#999;font-size:13px;margin-bottom:12px}\
         .cards{display:flex;flex-wrap:wrap;gap:12px}\
         .card{width:220px;background:#000;border-radius:6px;overflow:hidden;border:2px solid transparent}\
         .card.keep{border-color:#4caf50}\
         .card img{width:100%;height:160px;object-fit:cover;display:block;background:#333}\
         .card .info{padding:8px;font-size:11px;word-break:break-all;color:#ccc}\
         .card .badge{display:inline-block;background:#4caf50;color:#000;font-size:10px;\
             padding:1px 6px;border-radius:4px;margin-bottom:4px;font-weight:bold}\
         a{color:#8ab4f8}",
    );
    html.push_str("</style></head><body>");
    html.push_str("<h1>Similar Photos Report</h1>");
    html.push_str(&format!(
        "<div class=\"summary\">Scanned <b>{}</b> photos in {:?}<br>\
         Found <b>{}</b> similar group(s) covering <b>{}</b> files<br>\
         Estimated reclaimable space if you keep only the largest per group: <b>{}</b></div>",
        entries.len(),
        args.dir,
        duplicate_groups.len(),
        total_dup_files,
        format_bytes(reclaimable)
    ));

    for (gi, group) in duplicate_groups.iter().enumerate() {
        html.push_str(&format!(
            "<div class=\"group\"><h2>Group {} &mdash; {} photos</h2>\
             <div class=\"meta\">reclaimable: {}</div><div class=\"cards\">",
            gi + 1,
            group.len(),
            format_bytes(group_reclaim(&entries, group))
        ));

        for (mi, &idx) in group.iter().enumerate() {
            let e = &entries[idx];
            let keep_class = if mi == 0 { " keep" } else { "" };
            let badge = if mi == 0 {
                "<span class=\"badge\">KEEP (largest)</span><br>"
            } else {
                ""
            };
            let thumb_src = thumb_names
                .get(&idx)
                .cloned()
                .unwrap_or_else(|| "".to_string());
            html.push_str(&format!(
                "<div class=\"card{}\"><a href=\"{}\" target=\"_blank\">\
                 <img src=\"{}/{}\" loading=\"lazy\"></a>\
                 <div class=\"info\">{}{}<br>{} &times; {}<br>{}</div></div>",
                keep_class,
                file_url(&e.path),
                thumb_dir_name,
                thumb_src,
                badge,
                html_escape(&e.path.display().to_string()),
                e.width,
                e.height,
                format_bytes(e.size)
            ));
        }

        html.push_str("</div></div>");
    }

    html.push_str("</body></html>");

    std::fs::write(&output, html).expect("failed to write HTML report");
    println!("Report written to {:?}", output);
    println!("Thumbnails written to {:?}", thumb_dir);
}
