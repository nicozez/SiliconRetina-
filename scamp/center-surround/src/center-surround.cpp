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

#define EVENT_THRESHOLD 5

using namespace SCAMP5_PE;

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
//    auto display_og = vs_gui_add_display("display_og",0,0);
//    auto display_cs = vs_gui_add_display("display_cs",0,1);
//    auto display_raw_diff = vs_gui_add_display("display_raw_diff",0,2);
//    auto display_ON_events = vs_gui_add_display("display_ON_events",1,0);
//    auto display_OFF_events = vs_gui_add_display("display_OFF_events",1,1);

    auto display_ON_events = vs_gui_add_display("display_ON_events",0,0);
    auto display_OFF_events = vs_gui_add_display("display_OFF_events",0,1);

    vs_stopwatch stopwatch;
    stopwatch.reset();

	// Frame Loop
    while(1){
        vs_frame_loop_control();

        // capture a full-scale image in A
		scamp5_get_image(A, B);

		auto timestamp_start = stopwatch.get_usec();

        scamp5_kernel_begin();
        	SET(R1, R2);
        	gauss(B, A, 5);

        	CLR(R2);
        	gaussv(C, A, 5);

        	sub(D, B, C);

        	CLR(R1);
        	SET(R2);
        	gaussh(C, A, 5);

        	sub(E, A, C);
        	add(E, E, D);

        	// Load threshold value (we do it every time because of analog decay)
			scamp5_in(B, EVENT_THRESHOLD); // B = threshold

			// Find temporal frame diff
			sub(D, E, F);

        	// Find ON-events
			mov(C, D);               // C = D
			sub(C, C, B);            // C = C - B
			where(C);
			MOV(R7, FLAG);
			all();                   // Restore flag=1

        	// Find OFF-events
			neg(C, D);               // C = -D
			sub(C, C, B);            // C = C - B
			where(C);
			MOV(R8, FLAG);
			all();                   // Restore flag=1

			// Store current center-surround for future
        	mov(F, E);

		scamp5_kernel_end();

		auto timestamp_end = stopwatch.get_usec();

		// Ping to supervisor
		send_ping();

		if(vs_gui_is_on()){
			vs_post_text("FPS=%lu\n",1000000 / (timestamp_end - timestamp_start));
			// scamp5_output_image(A,display_og);
			// scamp5_output_image(E,display_cs);
			// scamp5_output_image(D,display_raw_diff);
			scamp5_output_image(R7,display_ON_events);
			scamp5_output_image(R8,display_OFF_events);
		}

    }

	return 0;
}
