TARGET = rtsp2display

SRC_PATH := $(CURDIR)

SRCS := $(foreach cf, $(SRC_PATH), $(wildcard $(cf)/*.c))
OBJS :=  $(SRCS:.c=.o)

LDFLAGS := -lspcdev -lavformat -lavcodec -lavutil
.PHONY: all clean

all: ${TARGET}

%.o:%.c
	@mkdir -p $(abspath $(dir $@))
	$(CC) -o $@ $(INCDIR) -c $<

$(TARGET):$(OBJS)
	@mkdir -p $(abspath $(dir $@))
	$(CC) -O3 -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -rf ${OBJS} ${TARGET}
