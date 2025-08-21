all:
	g++ -std=c++17 ofdm_sym.cpp -O2 -o ofdm_sym -Wall

clean:
	rm -rf ofdm_sym

