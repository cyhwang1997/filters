DIRS := build

all:
	@for i in $(DIRS); do $(MAKE) -C $$i $@; done

doc:
	$(MAKE) -C $@

test:
	@for i in $(DIRS); do $(MAKE) -C $$i $@; done

install:
	@for i in $(DIRS); do $(MAKE) -C $$i $@; done

uninstall:
	@for i in $(DIRS); do $(MAKE) -C $$i $@; done

clean:
	@for i in $(DIRS); do $(MAKE) -C $$i $@; done

distclean:
	rm -rf $(DIRS) Makefile

.PHONY: all doc test install uninstall clean distclean
