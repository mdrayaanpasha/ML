/* = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

                                🤵 Author: Rayaan Pasha
                                📂 File_Name: Kmeans.c
                                📅 Date_Of_Update: 26-02-24

                                📃 Description: the following c code is to compute
                                k-means algorithm from scratch.

= = = = = = = = = = = = = = = = = = = = = = = = = = == = = = = = = = = = = = = = */

#include<stdio.h>
#include<math.h>
#include<float.h>
#include<stdlib.h>

// = = = = = = = = = = = = = = ⚙️ PRE PROCESSORS = = = = = = = = = = = = = =
#define MAX_ITER 100
#define N 6
#define K 2

// = = = = = = = = = = = = = = 🏗️ NEW DATA TYPES = = = = = = = = = = = = = =
typedef struct{
    double x,y;
}Point;

// = = = = = = = = = = = = = = 🌍 GLOBAL VARIABLE = = = = = = = = = = = = = =
Point data[N] = {
    {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0},
    {8.0, 8.0}, {9.0, 9.0}, {10.0, 10.0}
};

Point Centroids[N];
int labels[N];

// = = = = = = = = = = = = = = 📜 PROTOTYPES = = = = = = = = = = = = = =
void Init_Centroids();
double Distance(Point x1,Point x2);
void Update_Centroids();
void Assign_Centroids();
int Check_Difference(Point Old_Centroids[]);
void kmeans();

// = = = = = = = = = = = = = = 🧑‍🚀 MAIN FUNCTION = = = = = = = = = = = = = =
int main(){
    kmeans();
    printClusters();
}

// = = = = = = = = = = = = = = 🔍 FUNCTION DEFINITION = = = = = = = = = = = = = =


void Init_Centroids(){
    /* = = = = = = = = = = = = = = = = = = = = = = = = = = =

    🎯 TO DO:
    - assign 1st k values to centroids as arbitrary values.
 
    = = = = = = = = = = = = = = = = = = = = = = = = = = = */
    for(int i=0;i<K;i++){
        Centroids[i]=data[i];
    }
}

double Distance(Point x1,Point x2){
    return sqrt(pow(x1.x-x2.x,2) + pow(x1.y-x2.y,2));
}


void Assign_Centroids(){
     /* = = = = = = = = = = = = = = = = = = = = = = = = = = =
     
     🎯 TO DO:
     - first assign 0 as the cluster, with min_dis=DBL_MAX
     - then go thru all centroids, and see which has least distance
     - assign that centroids as the cluster, by changing label.

    = = = = = = = = = = = = = = = = = = = = = = = = = = = */

    for(int i=0;i<N;i++){
        int Best_Cluster = 0;
        double Min_Distance = DBL_MAX;
        for(int j=0;j<K;j++){
            double d = Distance(data[i],Centroids[j]);
            if(d < Min_Distance){
                Min_Distance=d;
                Best_Cluster=j;
            }
        }
        labels[i]=Best_Cluster;
    }

}


void Update_Centroids(){
    /* = = = = = = = = = = = = = = = = = = = = = = = = = = =
     
     🎯 TO DO:
     - calculate the mean of each cluster, and then update, their centroid to that.

    = = = = = = = = = = = = = = = = = = = = = = = = = = = */
    Point New_Centroids[K] = {0};
    int count[K] = {0};

    for(int i=0;i<N;i++){
        int cluster = labels[i];
        New_Centroids[cluster].x+=data[i].x;
        New_Centroids[cluster].y+=data[i].y;
        count[cluster]++;
    }

    for(int i=0;i<K;i++){
        if(count[i] > 0){
            Centroids[i].x=New_Centroids[i].x/count[i];
            Centroids[i].y=New_Centroids[i].y/count[i];
        }
    }

}


int Check_Difference(Point Old_Centroids[]){
    /* = = = = = = = = = = = = = = = = = = = = = = = = = = =
     🎯 TO DO:
     - here im checking if there difference b/w any old centroid and new is not more then the magic number we are expecting, if so then 1, else 0.

    = = = = = = = = = = = = = = = = = = = = = = = = = = = */

    for(int i=0;i<K;i++){
        if(Distance(Old_Centroids[i],Centroids[i])>1e-4){
            return 1;
        }
    }
    return 0;
}




void kmeans(){
    
    /* = = = = = = = = = = = = = = = = = = = = = = = = = = =
    🎯 TO DO:
    - initilize centroids.
    - store old centroids.
    - assign clusters.
    - update clusters.
    
    if centroids not changed, then break. else continue.
    
    = = = = = = = = = = = = = = = = = = = = = = = = = = = */
    Init_Centroids();
    Point Old_Centroids[K];

    for(int i=0;i<MAX_ITER;i++){
        for (int j=0;j<K;j++) Old_Centroids[j] = Centroids[j];
        Assign_Centroids();
        Update_Centroids();
        if(!Check_Difference(Old_Centroids)) break;

    }
}

void printClusters() {
    /*
    🎯 TO DO:
    - traverse thru each data point and get thier label, i.e cluster.
    
    */
    for (int i = 0; i < N; i++) {
        printf("Point (%.2f, %.2f) -> Cluster %d\n", data[i].x, data[i].y, labels[i]);
    }
}
