TARGETS = test

CXX = g++

LD = g++

LDFLAGS += -lssl -lcrypto 

all: $(TARGETS)

test: test.cc

$(TARGETS):
	$(LD) $^ $(LDFLAGS) -o $@

clean:
	rm test
