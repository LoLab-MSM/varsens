// g++ -o SobolGen SobolGen.cpp -lQuantLib -I/usr/local/include/boost

#include <ql/math/randomnumbers/sobolrsg.hpp>
#include <iostream>

using namespace std;
using namespace QuantLib;

typedef std::vector<QuantLib::Real> QMCPoint;

int usage(int retval)
{
	puts("Usage: SobolGen dimensions samples <seed>");
	return retval;
}

int main (int argc, char* argv[])
{
	// Decode command line arguments, with check for validity
	char *pCh;
	if (argc != 3 && argc != 4) return(usage(1));
	unsigned long dimensions = strtoul(argv[1], &pCh, 10);
	if (pCh == argv[1] || *pCh != '\0')
	{
		puts("Invalid dimension");
		return(usage(2));
	}
	unsigned long samples    = strtoul(argv[2], &pCh, 10);
	if (pCh == argv[2] || *pCh != '\0')
	{
		puts("Invalid samples");
		return(usage(3));
	}
	unsigned long seed  = 64;
	if(argc >= 4)
	{
		seed = strtoul(argv[3], &pCh, 10);
		if (pCh == argv[3] || *pCh != '\0')
		{
			puts("Invalid seed");
			return(usage(4));
		}
	}
	
	// The main event
	// Constructor with seed for deterministic sequence, (dim, seed)
	SobolRsg sobol(dimensions, seed, SobolRsg::SobolLevitanLemieux);
	QMCPoint point; // For storing a single point

	sobol.skipTo(4096);// InitialSkip

	for(unsigned long j = 0; j<samples; ++j)
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
	return 0;
}
