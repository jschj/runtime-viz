.PHONY: zlib
zlib:
	mkdir -p libs/zlib/
	cd zlib; ./configure
	$(MAKE) -C zlib/
	cp zlib/libz.a libs/zlib/
	cp zlib/libz.so libs/zlib/
	cp zlib/libz.so.1 libs/zlib/
	cp zlib/libz.so.1.2.11 libs/zlib/
	cp zlib/zlib.h libs/zlib/
