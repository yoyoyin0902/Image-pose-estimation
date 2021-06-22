TARGET = orb
CFLAGS = -I/usr/local/include/opencv4/
CLIBS = -L/usr/local/lib/ -lopencv_core -lopencv_imgcodecs -lopencv_features2d -lopencv_highgui -lopencv_xfeatures2d -lopencv_flann

all: $(TARGET)
	

$(TARGET): orb.cpp
	g++ -DNDEBUG orb.cpp -o $(TARGET) $(CFLAGS) $(CLIBS)

clean:
	rm $(TARGET) 
