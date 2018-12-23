/**
  * micksort - quicksort on MPI
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

#include <emmintrin.h>
#include <pmmintrin.h>

typedef uint32_t elem_t;
typedef unsigned char map_t;
typedef map_t* mapptr_t;

//#define N RAND_MAX+1UL // Number of possible values obtained from rand()
#define N 2147483648UL
#define CHUNK_SIZE 268435456U 
#define N_CHUNKS ((N) / (CHUNK_SIZE))

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
printSeq( elem_t* buf, uint64_t nmemb ) {
    for( uint64_t i =0; i < nmemb; i++ )
        printf( "%d,", buf[i] );
    printf( "\n" );
}

static void
printMap( mapptr_t map ) {
    for( uint64_t i=0; i < N; i++ ) {
        if( map[i] ) 
            printf( "%ld(%d), ", i, map[i] );
    }
    printf( "\n" );
}

static void
randSeq( elem_t* buf, uint64_t nmemb, int seed, int rank ) {
    srand( SEED[seed] + rank );

    for( uint64_t i =0; i < nmemb; i++ )
        buf[i] = rand() /*% 100*/;
}

static void
randSeq2( mapptr_t map, uint64_t count, int seed, int rank ) {
    printf (  "Generating %ld prn's\n", count );
    srand_simple( SEED[seed] + rank );

    for( uint64_t i =0; i < count; i++ ) {
        elem_t num = rand_simple() % 20;
        map[num] ++;

    //    assert( map[num] < 255 );
    }

 /*   for( uint64_t c = 0; c < N_CHUNKS; c++ ) {
        srand( SEED[seed] + rank );
        
        uint64_t min = c    * CHUNK_SIZE;
        uint64_t max = (c+1)* CHUNK_SIZE;

        for( uint64_t i =0; i < count; i++ ) {
            uint64_t num = rand();
            if( num >= min && num < max )
                map[num] ++;
        }
        
    }*/

    printf( "done.\n" );
}

static void
merge( elem_t* buf, uint64_t m, uint64_t n ) {
    assert( m >= n );

   /* elem_t* tmp = calloc( m, sizeof(elem_t) );
    memcpy( tmp, buf+m, n * sizeof(elem_t) );

    uint64_t i=0, j=m+1, off =0;

    while( i < m && j < m+n ) {
        assert( off < j );
        if( i < m && ( tmp[i] < buf[j] || j >= m+n ) ) {
            buf[off++] = tmp[i++];
        } else {
            buf[off++] = buf[j++];
        }
    }

    free( tmp );*/
    uint64_t i=0, j=m+1, off =0;

    while( i < m && j < m+n ) {
        assert( i < j );
        if( j < m+n && ( buf[j] < buf[i] || i >= m ) ) {
            // Insert
            elem_t tmp =buf[j];
            memmove( buf + i + 1, buf + i, (j-i) * sizeof(elem_t) ); 
            buf[i] = tmp;
            j++;
        } else {
            // No action
        }
        i++;
    }
}

static void
merge2( mapptr_t map1, mapptr_t map2 ) {
    for( uint64_t i=0; i < N; i+=16 ) {
        //map1[i] += map2[i];
        //assert( map1[i] < 255 );
        __m128i a = _mm_lddqu_si128( (const __m128i*) &map1[i] );
        __m128i b = _mm_lddqu_si128( (const __m128i*) &map2[i] );
        a = _mm_add_epi8( a, b );
        _mm_storeu_si128( (__m128i*) &map1[i], a );
    }
}

static int
elem_compare( const void* apt, const void* bpt ) {
    elem_t a =*(elem_t*)apt;
    elem_t b =*(elem_t*)bpt;

    return a-b;
}

static uint64_t
psort( elem_t** buf, uint64_t* offset, int rank, int remote_rank, uint64_t m, uint64_t n, bool root ) {

    if( *offset == 0 ) {
        // First run, simple sort
        qsort( *buf, m, sizeof(elem_t), elem_compare );
    } else {
        // Need to merge two sorted sub-sequences
        merge( *buf, *offset, m - *offset );
        //qsort( *buf, m, sizeof(elem_t), elem_compare );
    }

    if( root ) return m;

    if( rank == remote_rank ) {
        *buf = realloc( *buf, (m + n) * sizeof(elem_t) );
        if( !*buf ) {
            printf( "Realloc error\n" );
        }
        MPI_Recv( (void*)(*buf+m), n, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

        return m+n;
    } else {
        MPI_Send( (void*)*buf, m, MPI_INT, remote_rank, 0, MPI_COMM_WORLD );
    }

    return m;
}

static uint64_t
reduce( mapptr_t map, int rank, int remote_rank, uint64_t m, uint64_t n, bool root ) {

    if( root ) return m;

    if( rank == remote_rank ) {
        mapptr_t tmp =calloc( N, sizeof(map_t) );
        if( !tmp ) {
            printf( "alloc error\n" );
        }
        MPI_Recv( (void*)tmp, N>>2, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

        merge2( map, tmp );

        free( tmp );
    } else {
        MPI_Send( (void*)map, N>>2, MPI_INT, remote_rank, 0, MPI_COMM_WORLD );
    }

    return m;
}
static uint64_t
partsize( uint64_t count, uint64_t ppn, uint64_t p, uint64_t np ) {
    uint64_t begin = p*ppn;
    uint64_t end = (p+1)*ppn;
    if( p==np-1 ) end = count;
    return end - begin;
}

static void
micksort( uint64_t count, int seed, int rank, int procs ) {
    
    elem_t* buf =NULL;
    uint64_t offset =0;
    uint64_t length =0;

    uint64_t nparts_div = (count/procs) * procs;
    for( uint64_t np =procs; np > 0; np = np/2 ) {
        uint64_t ppn = nparts_div / np; // Parts per node
        for( uint64_t p=0; p < np; p++ ) {
            uint64_t m =partsize( count, ppn, p, np );
            if( m<=0 ) continue;
            
            int src_rank = p * procs/np;
            int dst_rank = (p/2) * procs/np * 2;
            
            if( rank == src_rank ) {
                if( buf == NULL ) {
                    buf = calloc( m, sizeof(elem_t) );
                    if( !buf ) {
                        fprintf( stderr, "Calloc error\n" );
                    }
                    randSeq( buf, m, seed, rank );
                    //printf( "Seq %d: ", rank ); printSeq( buf, m );
                }

                uint64_t n = partsize( count, ppn, p + (src_rank==dst_rank?1:-1), np );

                //printf( "[level %d] Rank %d, sending to %d. Sorting %d elements, receiving %d elements...\n", 
                //        procs/np, src_rank, dst_rank, m, n );
                length =psort( &buf, &offset, src_rank, dst_rank, m, n, np==1?true:false );
            }
        }
        // Uncomment on weird segfaults in edge cases :)
        MPI_Barrier( MPI_COMM_WORLD );
    }

    /*if( rank == 0 ) {
        printf( "Result: " ); printSeq( buf, length );
    }*/
}

static void
micksort2( mapptr_t map, uint64_t count, int seed, int rank, int procs ) {
    
    memset( map, 0, N * sizeof(map_t) );

    uint64_t nparts_div = (count/procs) * procs;
    for( uint64_t np =procs; np > 0; np = np/2 ) {
        uint64_t ppn = nparts_div / np; // Parts per node
        for( uint64_t p=0; p < np; p++ ) {
            uint64_t m =partsize( count, ppn, p, np );
            if( m<=0 ) continue;
            
            int src_rank = p * procs/np;
            int dst_rank = (p/2) * procs/np * 2;
            
            if( rank == src_rank ) {
                if( np == procs ) {
                    randSeq2( map, m, seed, rank );
                    //printf( "Seq %d: ", rank ); printSeq( buf, m );
                }

                uint64_t n = partsize( count, ppn, p + (src_rank==dst_rank?1:-1), np );

                //printf( "[level %d] Rank %d, sending to %d. Sorting %d elements, receiving %d elements...\n", 
                //        procs/np, src_rank, dst_rank, m, n );
                reduce( map, src_rank, dst_rank, m, n, np==1?true:false );
            }
        }
        // Uncomment on weird segfaults in edge cases :)
        MPI_Barrier( MPI_COMM_WORLD );
    }

    /*if( rank == 0 ) {
        printf( "Result: " ); printSeq( buf, length );
    }*/
}

static void
micksort3( mapptr_t map, uint64_t count, int seed, int rank, int procs ) {

    /* Generation part */
    const size_t npp = count / procs; // Numbers to generate per process
    const size_t c = rank==0 ? count % procs : 0; // Remainder in case count % P != 0

    // Generate and sort the local sequence
    randSeq2( map, npp+c, seed, rank );

    printf( "Node %d: ", rank ); printMap( map );

    /* Communication part */
    const int bpp = N / procs; // Number of bins per process
    mapptr_t inmap = calloc( N, sizeof(map_t) );

/*    MPI_Request req[procs];
    MPI_Status stat[procs];

    for( int i =0; i < procs; i++ ) {
        if( i == rank ) continue;
        size_t off = bpp * i;
        MPI_Send( (void*)(map+off), bpp, MPI_BYTE, i, 0, MPI_COMM_WORLD );

        printf( "%d -> %d\n", rank, i );
    }


    size_t off =0;
    for( int i =0; i < procs; i++ ) {
        if( i == rank ) continue;
        MPI_Recv( (void*)(inmap+off), bpp, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        off +=bpp;
        printf( "%d <- %d\n", rank, i );
    }*/

    for( int root =0; root < procs; root++ )
        MPI_Gather( (void*)(map+root*bpp), bpp, MPI_BYTE, inmap, bpp, MPI_BYTE, root, MPI_COMM_WORLD );



}

int
main( int argc, char** argv ) {

    int procs, rank;
    int seed =0;
    //uint64_t count =16000000000UL; // for now
    //uint64_t count =80000000;
    uint64_t count =16;
/*    elem_t binSize = (1L << 31) / count;

    elem_t* buf = calloc( count, sizeof(elem_t) );
    srand( SEED[seed] + rank );

    uint64_t off =0;
    for( uint64_t i =0; i < count; i++ ) {
        elem_t x = rand();
        buf[off++] = x;
    }

    qsort( buf, count, sizeof(elem_t), elem_compare );

    elem_t max_dif =0;

    for( uint64_t i =0; i < count; i++ ) {
        int64_t binBase = binSize * (i+1);

        int64_t dif =(int64_t)binBase - (int64_t)buf[i];
        if( dif < 0 ) dif = -dif;


        if( (uint32_t)dif > max_dif ) max_dif =(uint32_t)dif;
    }

    printf ( "Stored %ld numbers. Max absolute delta is %d in bins of size %d\n", count, max_dif, binSize );*/

  /*  for( uint64_t i =0; i < count; i++ ) {
        int64_t binBase = binSize * (i+1);

        int64_t dif =(int64_t)binBase - (int64_t)buf[i];
        printf( "[%ld] bin %ld, value %ld (%d)\n", i, binBase, dif, buf[i] );

    }


    free( buf );*/

    MPI_Init( &argc, &argv );

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &procs );

    if(rank==0) printf( "Started.\n" );

    mapptr_t map =calloc( N, sizeof(map_t) );
    if( !map ) {
        fprintf( stderr, "alloc error\n" );
        return 1;
    }

    double start_time = MPI_Wtime();
    micksort3( map, count, seed, rank, procs );
    double end_time = MPI_Wtime();
    
    
    if( rank==0) printf( "(t) Sorted %ld elements in %gs\n", count, end_time-start_time );

   // if( rank==0) printMap( map );
    if( rank==0 ) {
        unsigned char max =0;
        for( size_t i =0; i < N; i++ ) {
            if( map[i] > max ) max =map[i];
        }

        printf( "Max count is %d\n", max );
    }

    free( map );
    
    MPI_Finalize();

    return 0;
}
