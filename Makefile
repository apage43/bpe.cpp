CXXFLAGS += -std=c++11 -g -Wall -Werror
CXXFLAGS += $(shell pkg-config --static --libs --cflags icu-i18n)

ttok: ttok.cpp
