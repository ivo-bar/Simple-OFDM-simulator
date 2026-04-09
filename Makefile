all:
	g++ -std=c++17 ofdm_sym.cpp -O2 -o ofdm_sym -Wall

new:
	g++ -std=c++17 ofdm_sym_claude_update.cpp -O2 -o ofdm_sym_new -Wall

clean:
	rm -rf ofdm_sym

