// g++ -o SobolGen SobolGen.cpp -lQuantLib -I/usr/local/include/boost

#include <ql/math/randomnumbers/sobolrsg.hpp>
#include <iostream>

using namespace std;
using namespace QuantLib;

typedef std::vector<QuantLib::Real> QMCPoint;

int main (int argc, char* argv[])
{
	// Constructor with seed for deterministic sequence, (dim, seed)
	SobolRsg sobol(106, 64, SobolRsg::SobolLevitanLemieux);
	QMCPoint point; // For storing a single point

	sobol.skipTo(4096);// InitialSkip

	for(int j = 0; j<40000; ++j)
	{
		point = sobol.nextSequence().value; // Generate Next Point of Sobol Sequence
		QMCPoint::size_type nextToLast = point.size() - 1;

		for(QMCPoint::size_type i = 0; i <= nextToLast; ++i)
		{
			cout << point[i];
			if(i != nextToLast) cout << ",";
		}
		cout << endl;
	} 
	return 0 ;
}
