CXXFLAGS += -std=c++11 -g -Wall -Werror
CXXFLAGS += $(shell pkg-config --static --cflags icu-i18n)
LDFLAGS += $(shell pkg-config --static --libs icu-i18n)

testtok: testtok.cpp bpe.o

clean:
	rm -f ttok bpe.o
