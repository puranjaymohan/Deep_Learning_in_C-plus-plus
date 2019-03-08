#include <iostream>
//#include <vector>

#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;
using namespace std;
int main()
{
	network<sequential> net;
	net << fc<sigmoid>(2,3) 
		<< fc<sigmoid>(3,1); 

	std::vector<vec_t> trainIn = {{0,0}, {0,1}, {1,0}, {1,1}};
	std::vector<vec_t> trainOut = {{0}, {1}, {1}, {0}};

	adam optimizer;
	net.fit<mse>(optimizer, trainIn, trainOut, 1, 15000);
	double loss = net.get_loss<mse>(trainIn, trainOut);
	cout << "Mean Squared Error : " << loss << endl;
	net.save("net");
	std::cout <<"Predicted Output :-"<<endl;
	std::cout <<"0 | 0 | "<< net.predict({0,0})[0] << std::endl;
	std::cout <<"0 | 1 | "<< net.predict({0,1})[0] << std::endl;
	std::cout <<"1 | 0 | "<< net.predict({1,0})[0] << std::endl;
	std::cout <<"1 | 1 | "<< net.predict({1,1})[0] << std::endl;
	return 0;
}
