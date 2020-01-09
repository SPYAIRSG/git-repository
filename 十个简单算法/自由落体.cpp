//一球从100米高度自由落下，
//每次落地后反跳回原高度的一半；再落下，求它在第10次落地时，共经过多少米？第10次反弹多高？
#include<stdio.h>
void fall01(){
	float sum=100.0;
	float fall_sum[11]={0.0};
	
	fall_sum[1]=100.0;
	fall_sum[2]=200.0;

	for(int i=3;i<11;i++){
			fall_sum[i]=fall_sum[i-1]+(fall_sum[i-1]-fall_sum[i-2])/2.0;
			//printf("%4d",fall_sum[i]);
	}
	
		printf("第十次落地时，共经过%4.1f米\n",fall_sum[10]);
		printf("第十次反弹%4.1f米",(fall_sum[10]-fall_sum[9])/4.0) ;
}
void fall02(){
	float sn=100.0,hn=sn/2.0;
	int i;
	for(i=2;i<=10;i++){
		sn=sn+hn*2;  //第n次落地时共经过的米数 
		hn=hn/2;  //	第n次反跳高度 
	}
	
		printf("第十次落地时，共经过%4.1f米\n",sn);
		printf("第十次反弹%4.1f米",hn);
}
int main(){
	fall01();
	return 0;
}

