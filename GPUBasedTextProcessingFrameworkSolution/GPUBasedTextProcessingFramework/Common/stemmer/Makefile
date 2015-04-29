cc = g++
INCLUDES = -I.

all : libstem_api.a

#one cpp file generate one .o file!!!
%.o : %.cpp
	cc -c -o $@ $< $(INCLUDES)

#just archive all .o files to a .a file. Then this .a file can be linked by extern program to gen executable file.
libstem_api.a : stem_api.o libstemmer.o api.o stem_UTF_8_english.o stem_ISO_8859_1_english.o utilities.o
	ar -rcs $@ $^


.PHONY clean:
	rm -f *.o
