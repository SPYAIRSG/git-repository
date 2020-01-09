//一只猴子摘了N个桃子第一天吃了一半又多吃了一个,第二天又吃了余下的
//一半又多吃了一个,到第十天的时候发现还有一个.
#include<stdio.h>

void h(){
	int n=1;
	for(int i=0;i<9;i++){
		n=(n+1)*2;
	}
	printf("第一天一共摘了%d个桃子",n);
}
int main(){
	h();
	return 0;
}
