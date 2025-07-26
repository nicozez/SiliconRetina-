#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <thread>
#include <chrono>

#pragma comment(lib, "ws2_32.lib")

SOCKET ping_sock = INVALID_SOCKET;

void start_ping_socket(const char* host, int port) {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed\n";
        return;
    }

    ping_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ping_sock == INVALID_SOCKET) {
        std::cerr << "Socket creation failed\n";
        WSACleanup();
        return;
    }

    sockaddr_in server_addr {};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = inet_addr(host); // e.g., 127.0.0.1

    if (connect(ping_sock, (SOCKADDR*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
        std::cerr << "Failed to connect to supervisor\n";
        closesocket(ping_sock);
        WSACleanup();
        ping_sock = INVALID_SOCKET;
        return;
    }

    std::cout << "Connected to supervisor on " << host << ":" << port << "\n";
}

void send_ping() {
    if (ping_sock != INVALID_SOCKET) {
        const char* msg = "ping\n";
        int sent = send(ping_sock, msg, strlen(msg), 0);
        if (sent == SOCKET_ERROR) {
            std::cerr << "Ping send failed\n";
            closesocket(ping_sock);
            WSACleanup();
            ping_sock = INVALID_SOCKET;
        }
    }
}

#include <scamp5.hpp>
#include <scamp5_all_algorithms.hpp>
#include <scamp5_all_templates.hpp>

using namespace SCAMP5_PE;

#define horizontal_fold(X) \
    mov(A, X);             \
    divq(X, A);            \
    bus(NEWS, X);          \
    bus(NEWS, XW);         \
    res(A);                \
    bus(A, X, NEWS);       \
    bus(X, A);             \

#define vertical_fold(X)   \
    mov(A, X);             \
    divq(X, A);            \
    bus(NEWS, X);          \
    bus(NEWS, XN);         \
    res(A);                \
    bus(A, X, NEWS);       \
    bus(X, A);             \

#define fold2(X)           \
	horizontal_fold(X);    \
	vertical_fold(X);      \

#define align_fold2_to_fold4(X)                \
	scamp5_load_pattern(R5, 3, 0, 252, 255);   \
    NOT(R5);                                   \
    movx(X, X, north);                         \
    scamp5_load_pattern(R5, 0, 0, 255, 252);   \
    NOT(R6);                                   \
    mov(X, X, west);                           \

#define fold4(X)                   \
	fold2(X);                      \
    align_fold2_to_fold4(X);       \
	fold2(X);                      \


int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: client.exe <port>\n";
        return 1;
    }

    const char* host = "127.0.0.1";
    int port = std::stoi(argv[1]);

    start_ping_socket(host, port);

	// Initialization
	vs_init();
    vs_on_shutdown([&](){
    	vs_post_text("M0 shutdown\n");
    });

	vs_gui_set_info(VS_M0_PROJECT_INFO_STRING);
    auto display_og = vs_gui_add_display("display_og",0,0);
    auto display_A = vs_gui_add_display("display_A",1,0);
    auto display_B = vs_gui_add_display("display_B",1,1);
    auto display_C = vs_gui_add_display("display_C",1,2);
    auto display_D = vs_gui_add_display("display_D",2,0);
    auto display_E = vs_gui_add_display("display_E",2,1);
    auto display_F = vs_gui_add_display("display_F",2,2);

    vs_stopwatch stopwatch;
    stopwatch.reset();

    // capture a full-scale image in F
    // F represents previous frame
	// We want to instantiate both current and previous frames at the start
    // to ensure there are no initial bursts of events
    scamp5_get_image(F, B);

	// Frame Loop
    while(1){
        vs_frame_loop_control();

        // capture a full-scale image in A and F
        // A represents current frame
		scamp5_get_image(A, B);

		// Readout original image
		scamp5_output_image(A, display_og);

		// For time tracking and FPS calculations
		auto timestamp_start = stopwatch.get_usec();

        scamp5_kernel_begin();

            // Compute inverse directional shifts
            subx(B, F, south, A);          // B = F_south - A | Inverse upward movement
            subx(C, F, east, A);           // C = F_east - A  | Inverse leftward movement
            subx(D, F, north, A);          // D = F_north - A | Inverse downward movement
            subx(E, F, west, A);           // D = F_west - A  | Inverse rightward movement

        	// Store current frame as previous for next compute cycle
        	mov(F, A);

            // Invert shifts to find correct ones
            abs(A, B);
            mov(B, A);

            abs(A, C);
            mov(C, A);

            abs(A, D);
            mov(D, A);

            abs(A, E);
            mov(E, A);

            // Fold 2x2
            fold4(B);
            fold2(C);
            // fold2(D);
            horizontal_fold(D);
            // vertical_fold(D);
            fold2(E);

            // Bottom right pixels in 4x4 mask
            scamp5_load_pattern(R5, 3, 0, 252, 252);
            NOT(R5);
            WHERE(R5);
            res(B);
            res(C);
            // res(D);
            res(E);
            ALL();


		scamp5_kernel_end();

		// For time tracking and FPS calculations
		auto timestamp_end = stopwatch.get_usec();

		// Ping to supervisor
		send_ping();

		if(vs_gui_is_on()){
			vs_post_text("FPS=%lu\n",1000000 / (timestamp_end - timestamp_start));

			scamp5_output_image(A, display_A);
			scamp5_output_image(B, display_B);
			scamp5_output_image(C, display_C);
			scamp5_output_image(D, display_D);
			scamp5_output_image(E, display_E);
			scamp5_output_image(F, display_F);
		}

    }

	return 0;
}
