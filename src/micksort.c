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

typedef uint32_t elem_t;

#define N_SEEDS 5
static const uint32_t SEED[N_SEEDS] = { 0x1234abcd, 0x10203040, 0x40e8c724, 0x79cbba1d, 0xac7bd459 };

static void
printSeq( elem_t* buf, size_t nmemb ) {
    for( size_t i =0; i < nmemb; i++ )
        printf( "%d,", buf[i] );
    printf( "\n" );
}

static void
randSeq( elem_t* buf, size_t nmemb, int seed, int rank ) {
    srand( SEED[seed] + rank );

    for( size_t i =0; i < nmemb; i++ )
        buf[i] = rand() /*% 100*/;
}

static void
merge( elem_t* buf, size_t m, size_t n ) {
    assert( m >= n );

    elem_t* tmp = calloc( m, sizeof(elem_t) );
    memcpy( tmp, buf+m, n * sizeof(elem_t) );

    size_t i=0, j=m+1, off =0;

    while( i < m && j < m+n ) {
        assert( off < j );
        if( i < m && ( tmp[i] < buf[j] || j >= m+n ) ) {
            buf[off++] = tmp[i++];
        } else {
            buf[off++] = buf[j++];
        }
    }

    free( tmp );
    /*size_t i=0, j=m+1, off =0;

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
    }*/
}

static int
elem_compare( const void* apt, const void* bpt ) {
    elem_t a =*(elem_t*)apt;
    elem_t b =*(elem_t*)bpt;

    return a-b;
}

static size_t
psort( elem_t** buf, size_t* offset, int rank, int remote_rank, size_t m, size_t n, bool root ) {

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
        MPI_Recv( (void*)(*buf+m), n, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

        return m+n;
    } else {
        MPI_Send( (void*)*buf, m, MPI_INT, remote_rank, 0, MPI_COMM_WORLD );
    }

    return m;
}

static int
partsize( int count, int ppn, int p, int np ) {
    int begin = p*ppn;
    int end = (p+1)*ppn;
    if( p==np-1 ) end = count;
    return end - begin;
}

static void
micksort( size_t count, int seed, int rank, int procs ) {
    
    elem_t* buf =NULL;
    size_t offset =0;
    size_t length =0;

    int nparts_div = (count/procs) * procs;
    for( int np =procs; np > 0; np = np/2 ) {
        int ppn = nparts_div / np; // Parts per node
        for( int p=0; p < np; p++ ) {
            int m =partsize( count, ppn, p, np );
            if( m<=0 ) continue;
            
            int src_rank = p * procs/np;
            int dst_rank = (p/2) * procs/np * 2;
            
            if( rank == src_rank ) {
                if( buf == NULL ) {
                    buf = calloc( m, sizeof(elem_t) );
                    randSeq( buf, m, seed, rank );
                    //printf( "Seq %d: ", rank ); printSeq( buf, m );
                }

                int n = partsize( count, ppn, p + (src_rank==dst_rank?1:-1), np );

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


int
main( int argc, char** argv ) {

    int procs, rank;
    int seed =0;
    size_t count =80000000; // for now
/*    elem_t binSize = (1L << 31) / count;

    elem_t* buf = calloc( count, sizeof(elem_t) );
    srand( SEED[seed] + rank );

    size_t off =0;
    for( size_t i =0; i < count; i++ ) {
        elem_t x = rand();
        buf[off++] = x;
    }

    qsort( buf, count, sizeof(elem_t), elem_compare );

    elem_t max_dif =0;

    for( size_t i =0; i < count; i++ ) {
        int64_t binBase = binSize * (i+1);

        int64_t dif =(int64_t)binBase - (int64_t)buf[i];
        if( dif < 0 ) dif = -dif;


        if( (uint32_t)dif > max_dif ) max_dif =(uint32_t)dif;
    }

    printf ( "Stored %ld numbers. Max absolute delta is %d in bins of size %d\n", count, max_dif, binSize );*/

  /*  for( size_t i =0; i < count; i++ ) {
        int64_t binBase = binSize * (i+1);

        int64_t dif =(int64_t)binBase - (int64_t)buf[i];
        printf( "[%ld] bin %ld, value %ld (%d)\n", i, binBase, dif, buf[i] );

    }


    free( buf );*/

    MPI_Init( &argc, &argv );

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &procs );

    double start_time = MPI_Wtime();
    micksort( count, seed, rank, procs );
    double end_time = MPI_Wtime();
    
    
    if( rank==0) printf( "(t) Sorted %ld elements in %gs\n", count, end_time-start_time );
    
    MPI_Finalize();

    return 0;
}
