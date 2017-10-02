CROSS_COMPILE = ../toolchain/bin/aarch64-linux-gnu-
CC	= @echo "CC $@"; $(CROSS_COMPILE)g++
LD	= @echo "LD $@"; $(CROSS_COMPILE)ld"
AR	= @echo "AR $@"; $(CROSS_COMPILE)ar"
STRIP = @echp "STRIP $@"; $(CROSS_COMPILE)strip"

ABSOLUTE_DIR = ../thirdpartylib

INCLUDE = -I $(ABSOLUTE_DIR)/include

CFLAGS += -I./ -O3 -fPIC -shared -std=c++11 $(INCLUDE) 

LINK_LIBS += -L $(ABSOLUTE_DIR)/lib -lopencv_core -lopencv_imgproc

TK_TEST_OBJS += autel_tk.o tk_api.o fhog.o tk_interface.o BoundaryDissimilarityMap.o rectangle.o

.PHONY : clean all

all: $(TK_TEST_OBJS)
	$(CC) $(CFLAGS) $(TK_TEST_OBJS) $(LINK_LIBS) -o libtracker.so

clean:
	@rm -rf $(TK_TEST_OBJS) 
	@echo "clean object done!"
