BIN := simphoto
BUILD_DIR := build
LINUX_TARGET := x86_64-unknown-linux-gnu
WINDOWS_TARGET := x86_64-pc-windows-gnu

.PHONY: all linux windows clean

all: linux windows

linux:
	cargo build --release --target $(LINUX_TARGET)
	mkdir -p $(BUILD_DIR)
	cp target/$(LINUX_TARGET)/release/$(BIN) $(BUILD_DIR)/$(BIN)-linux-x86_64

windows:
	cargo build --release --target $(WINDOWS_TARGET)
	mkdir -p $(BUILD_DIR)
	cp target/$(WINDOWS_TARGET)/release/$(BIN).exe $(BUILD_DIR)/$(BIN)-windows-x86_64.exe

clean:
	cargo clean
	rm -rf $(BUILD_DIR)
