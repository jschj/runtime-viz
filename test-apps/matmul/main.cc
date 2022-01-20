#include <iostream>
#include <iomanip>

#include "matmul.h"
#include "test.h"

int main(int argc, char **argv)
{
	std::cout << "PMPP Hello World!" << std::endl;

	int n = argc >= 2 ? std::stoi(argv[1]) : -1;
	pmpp::load(n);

	print_cuda_devices();
	std::cout << std::setprecision(10);
	matmul();
}
