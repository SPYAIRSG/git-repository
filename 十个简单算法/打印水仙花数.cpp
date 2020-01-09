#include<stdio.h>
void print_flowerNumber(){
	int a,b,c;
	for(int i=100;i<=999;i++){
		a=i/100;
		b=(i/10)%10;
		c=(i%100)%10;
		if(a*a*a+b*b*b+c*c*c==i){
			printf("%4d",i);
		} 
	}
}
int main(){
	print_flowerNumber();
	return 0;
}
