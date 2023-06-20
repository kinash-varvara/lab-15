#include <iostream>
#include <chrono>
#include "matrix.h"

int main() {

    for (int i = 2; i < 3860; i += 50) {
        Matrix<int> m1(i, i);
        int temp1 = rand();
        m1.fill(temp1);

        Matrix<int> m2(i, i);
        int temp2 = rand();
        m2.fill(temp2);

        auto start = std::chrono::system_clock::now();

        Matrix<int> m_res = m1 + m2;

        auto end = std::chrono::system_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        //std::cout << m_res;
        std::cout <<duration.count()<< std::endl;
    }


    //auto start = std::chrono::system_clock::now();





    return 0;
}