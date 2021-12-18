#include "matrix.h"
#include "mul_cpu.h"
#include "common.h"

void matrix_mul_cpu(const CPUMatrix &m, const CPUMatrix &n, CPUMatrix &p)
{
	// TODO: Task 3

	for (int i = 0; i < p.height; ++i) {
		for (int j = 0; j < p.width; ++j) {
			float dot = 0;

			for (int k = 0; k < m.width; ++k)
				dot += m.elements[i * m.width + k] * n.elements[k * n.width + j];

			p.elements[i * p.width + j] = dot;
		}
	}
}
