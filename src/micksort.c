/**
  * micksort - counting sort on MPI
  *
  * (C) 2018 by Micky Faas, LIACS Leiden
  * micky<at>eduktity<dot>org
  */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

#include <emmintrin.h>
#include <pmmintrin.h>

#define FAST_RAND     // Use the old-fashioned POSIX PRNG instead of the new one from (G)LIBC
#define HALF_BINS     // Use bins of 4-bits instead of 8 to save memory and improve cache performance

typedef uint8_t map_t;
typedef map_t* mapptr_t;

//#define N RAND_MAX+1UL // Number of possible values obtained from rand()
#define N 2147483648UL
//#define N 64

#ifdef HALF_BINS
#define BUF_SIZE 1073741824UL
#define BINS_PER_BYTE 2
#else
#define BUF_SIZE N
#define BINS_PER_BYTE 1
#endif

#define N_SEEDS 5
static const uint32_t SEED[N_SEEDS] = { 0x1234abcd, 0x10203040, 0x40e8c724, 0x79cbba1d, 0xac7bd459 };

static uint32_t RSTATE =0;

static void 
srand_simple( uint32_t seed ) {
    if( seed == 0 ) RSTATE =1;
    RSTATE =seed;
}

static inline uint32_t
rand_simple() {
    // Copied from the GLIBC source code for TYPE0 PRNGs
    RSTATE = ((RSTATE * 1103515245U) + 12345U) & 0x7fffffff;
    return RSTATE;
}

static void
printMap( mapptr_t map, size_t n, int base ) {
    for( uint64_t i=0; i < n; i++ ) {
        if( map[i] ) 
            printf( "%ld(%d), ", i+base, map[i] );
    }
    printf( "\n" );
}

static void
randSeq2( mapptr_t map, uint64_t count, int seed, int rank ) {
    memset( map, 0, BUF_SIZE * sizeof(map_t) );
#ifdef FAST_RAND
    srand_simple( SEED[seed] + rank );
#else
    srand( SEED[seed] + rank );
#endif

    for( uint64_t i =0; i < count; i++ ) {
#ifdef FAST_RAND
        uint32_t num = rand_simple() % 64;
#else        
        uint32_t num = rand()% 64;
#endif

#ifdef HALF_BINS
        uint32_t idx =(num & 0x7FFFFFFE) >> 1;  // Remove the least bit, divide by two
        uint8_t f = map[idx];                   // Get the corresponding pair of bins
        uint32_t pos =num & 0x1;                // Get the least bit to determine which pair to increment
        uint8_t mask =0xF << (pos ? 4 : 0);    // Mask the pair we want
                                                // Write back the other pair + our pair incremented by one
        f = (f & ~mask) | ((f & mask) + (1 << (pos ? 4 : 0)));
        map[idx] =f;

        assert( (f & mask) != 0 );              // Check for overflow
#else
        map[num]++;
        assert( map[num] != 0 );
#endif
    }
}

static void
mergemm( mapptr_t restrict dst, mapptr_t restrict maps, int n, int stride ) {

    // The following is the multi-core SIMD variant of the commented code.

    /*for( int i =0; i < stride; i++ ) {
        for( int j =0; j < n; j++ ) {
            dst[i] += maps[j*stride+i];
        }
    }*/
    const blockSize = stride / 64;

#pragma omp parallel for
    for( int b =0; b < stride; b+= blockSize ) {
        for( int i =0; i < blockSize; i+=16 ) {
            __m128i sum = _mm_setzero_si128();
            for( int j =0; j < n; j++ ) {
                __m128i a = _mm_load_si128( (const __m128i*) &maps[j*stride+i+b] );
                sum = _mm_add_epi8( sum, a );
            }
            _mm_store_si128( (__m128i*) &dst[i+b], sum );

        }
    }
}

static mapptr_t
micksort3( uint64_t count, int seed, int rank, int procs ) {

    /* Generation part */
    const size_t npp = count / procs; // Numbers to generate per process
    const size_t c = rank==0 ? count % procs : 0; // Remainder in case count % P != 0

    // Generate and sort the local sequence
    mapptr_t map = _mm_malloc( BUF_SIZE*sizeof(map_t), 16 );

    randSeq2( map, npp+c, seed, rank );

    //printf( "Node %d: ", rank ); printMap( map, N, 0 );

    /* Communication part */
    const int bpp = N / procs; // Number of bins per process
    const int base = rank * bpp; 

    mapptr_t sorted;

    if( procs > 1 ) {
        mapptr_t inmap = _mm_malloc( BUF_SIZE * sizeof(map_t), 16 );
    
        MPI_Request reqs[procs];

        for( int root =0; root < procs; root++ )
            MPI_Igather( map+root*(bpp/BINS_PER_BYTE), 
                         bpp/BINS_PER_BYTE, 
                         MPI_BYTE, 
                         inmap, 
                         bpp/BINS_PER_BYTE, 
                         MPI_BYTE, 
                         root, MPI_COMM_WORLD, reqs+root );

        MPI_Waitall( procs, reqs, MPI_STATUSES_IGNORE );

        _mm_free( map );

        sorted =_mm_malloc( (bpp/BINS_PER_BYTE)*sizeof(map_t), 16 );

        mergemm( sorted, inmap, procs, bpp/BINS_PER_BYTE );

        _mm_free( inmap );

    }
    else sorted = map;

    return sorted;
}

void
printParallel( mapptr_t restrict map, unsigned int skip, int rank, int procs ) {
    const unsigned int bpp = N / procs;    // Number of bins per process
    const unsigned int base = rank * bpp;  // Value of the first element in the map
    unsigned int skipCount =0;             // Number of elements skipped so far


    for( int r =0; r < procs; r++ ) {
        if( rank == r ) {
            if( rank != 0 ) {
                MPI_Recv( &skipCount, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            }

            for( unsigned int i =0; i < bpp/BINS_PER_BYTE; i++ ) {

                uint8_t f = map[i];   // Get the corresponding pair of bins

#ifdef HALF_BINS
                int bins[2] = { f & 0xF, (f >> 4) & 0xF };
#else
                int bins[1] = { f };
#endif

                for( int b =0; b < BINS_PER_BYTE; b++ ) {
                    for( unsigned int j =0; j < bins[b]; j++ ) {
                        if( skipCount == 0 ) {
                            printf( "%d ", base + i*BINS_PER_BYTE + b );
                        }
                        if( ++skipCount == skip ) skipCount =0;
                    }
                }
            }

            if( rank != procs - 1 )
                MPI_Send( &skipCount, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD );
            else
                printf( "\n" );

            fflush( stdout );

        }
        MPI_Barrier( MPI_COMM_WORLD );
    }
}

int
main( int argc, char** argv ) {

    int 	    procs, rank;
    int 	    seed =0, skip =1;
    uint64_t 	    count = 0;
    int             c;
    const char    * short_opt = "hs:n:i:";
    struct option   long_opt[] =
    {
	{"help",          no_argument,       NULL, 'h'},
	{"count",         required_argument, NULL, 'n'},
	{"seed",          required_argument, NULL, 's'},
	{"skip",          required_argument, NULL, 'i'},
	{NULL,            0,                 NULL, 0  }
    };

    while( (c = getopt_long( argc, argv, short_opt, long_opt, NULL )) != -1 )
    {
	switch( c )
	{
	    case -1:
	    case 0:
		break;

	    case 's':
		seed = atoi( optarg );
                if( seed < 0 || seed >= N_SEEDS ) { 
                    fprintf( stderr, "(e) Invalid seed index.\n" );
                    return -2;
                }
		break;

            case 'n':
                count =atol( optarg );
                break;

            case 'i':
                skip =atoi( optarg );
                break;

	    case 'h':
		printf( "(i) Usage: %s [OPTIONS]\n", argv[0] );
		printf( "  -s, --seed n              seed index for pseudo random numbers 0..%d\n", N_SEEDS-1 );
		printf( "  -n, --count n             total length of the random array\n" );
		printf( "  -i, --skip n              print only every n-th number\n" );
		printf( "  -h, --help                print this help and exit\n" );
		printf( "\n");
		return 0;

	    case ':':
	    case '?':
		fprintf( stderr, "(e) Try `%s --help' for more information.\n", argv[0] );
		return -2;

	    default:
		fprintf( stderr, "(e) %s: invalid option -- %c\n", argv[0], c );
		fprintf( stderr, "(e) Try `%s --help' for more information.\n", argv[0] );
		return -2;
	};
    };

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &procs );

#ifdef FAST_RAND
    const char* frstr = "ON";
#else
    const char* frstr = "OFF";
#endif

    if(rank==0) printf( "(i) Started.\n(i) Sorting %ld numbers on %d processes.\n(i) Seed is 0x%x\n(i) Fast random is %s\n(i) Bins are %d bits\n",
            count, procs, SEED[seed], frstr, 8 / BINS_PER_BYTE );

    double start_time = MPI_Wtime();
    
    mapptr_t map = micksort3( count, seed, rank, procs );
    
    double end_time = MPI_Wtime();
   
    if( skip != 0 )
        printParallel( map, skip, rank, procs );
    
    if( rank==0) printf( "(t) Sorting took %gs\n", count, end_time-start_time );


    // Some stuff for debugging
    // if( rank==0) printMap( map );
/*    if( rank==0 ) {
        unsigned char max =0;
        for( size_t i =0; i < N; i++ ) {
            if( map[i] > max ) max =map[i];
        }

        printf( "Max count is %d\n", max );
    }*/

    _mm_free( map );
    
    MPI_Finalize();

    return 0;
}
