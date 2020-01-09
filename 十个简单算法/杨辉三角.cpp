#include<stdio.h>
void YanghuiTriangle(){
	int i,j,triangle[8][8]={0};
	for(i=0;i<8;i++){
		for(int j=0;j<=i;j++){
			triangle[i][j]=1;
			//printf("%4d",triangle[i][j]);
		}
		//printf("\n");
	} 
	for(i=2;i<8;i++){
		for(j=1;j<i;j++){
			triangle[i][j]=triangle[i-1][j]+triangle[i-1][j-1];
		}
	}
	for(i=0;i<8;i++){
		for(int j=0;j<=i;j++){
		
			printf("%4d",triangle[i][j]);
		}
		printf("\n");
	} 
	
	
}
int main(){
	YanghuiTriangle();
	return 0;
}
