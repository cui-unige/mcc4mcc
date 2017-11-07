# Programs for processing
LEX				= flex
YACC			= bison
CC				= g++
CUT				= cut

# Compiler and linker flags
CFLAGS			= -std=c++14 -I.
LDFLAGS			= -llpsolve55 -static

# Input files
FLEX_SOURCES	= $(shell find * -name "*.l")
BISON_SOURCES	= $(shell find * -name "*.y")
SOURCES			= $(shell find * -name "*.cpp" | grep -v ".\\(parser\\|lexer\\).cpp")		\
				  $(BISON_SOURCES:.y=.parser.cpp)											\
				  $(FLEX_SOURCES:.l=.lexer.cpp)
OBJECTS			= $(SOURCES:.cpp=.o)
TARGET			= verifypn-linux64

include rules.mk
