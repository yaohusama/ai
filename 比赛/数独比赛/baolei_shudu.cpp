//<pre code_snippet_id="1592833" snippet_file_name="blog_20160301_1_4048211" name="code" class="cpp">#include <iostream>
#include<iostream> 
#include<cstdio>
using namespace std;
int arr[9][9];
/* 构造完成标志 */
bool sign = false;
//以下是在灰格不相邻，灰格和其他带有数字的都有白格相间隔。 
/* 创建数独矩阵 */
int num[9][9];
 
/* 函数声明 */
void Input();
void Output();
bool Check(int n, int key);
int DFS(int n);
int siz;
int lix[100];
int liy[100];
/* 主函数 */
int main()
{
    cout << "请输入一个9*9的数独矩阵，空位以0表示:" << endl;
    Input();
    DFS(0);


}
 
/* 读入数独矩阵 */
void Input()
{
	cin>>siz;
	for(int i=0;i<siz;i++){
		cin>>lix[i]>>liy[i];
	}
    char temp[9][9];
    for (int i = 0; i < 9; i++)
    {
        for (int j = 0; j < 9; j++)
        {
//            scanf("%s",&temp[i][j]);
cin>>temp[i][j];
            num[i][j] = temp[i][j] - '0';
        }
    }
}
 
/* 输出数独矩阵 */
void Output()
{
    cout << endl;
    for (int i = 0; i < 9; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            cout << num[i][j] << " ";
            if (j % 3 == 2)
            {
                cout << "   ";
            }
        }
        cout << endl;
        if (i % 3 == 2)
        {
            cout << endl;
        }
    }
}
 
/* 判断key填入n时是否满足条件 */
bool Check(int n, int key)
{
	int xx=n/9; int yy=n%9;
			for(int i=0;i<siz;i++){
			if(xx==lix[i]&&yy==liy[i]){
				if(xx>0&&num[xx-1][yy]>=key) return false;
				if(yy>0&&num[xx][yy-1]>=key) return false;
			}
		}
	if(xx>0){
		int x=xx-1;int y=yy;
		for(int i=0;i<siz;i++){
			if(x==lix[i]&&y==liy[i]){
				if(num[x][y]<=key) return false;
			}
		}
	} 
		if(yy>0){
		int x=xx;int y=yy-1;
		for(int i=0;i<siz;i++){
			if(x==lix[i]&&y==liy[i]){
				if(num[x][y]<=key) return false;
			}
		}
	} 
    /* 判断n所在横列是否合法 */
    for (int i = 0; i < 9; i++)
    {
        /* j为n竖坐标 */
        int j = n / 9;
        if (num[j][i] == key) return false;
    }
 
    /* 判断n所在竖列是否合法 */
    for (int i = 0; i < 9; i++)
    {
        /* j为n横坐标 */
        int j = n % 9;
        if (num[i][j] == key) return false;
    }
 
    /* x为n所在的小九宫格左顶点竖坐标 */
    int x = n / 9 / 3 * 3;
 
    /* y为n所在的小九宫格左顶点横坐标 */
    int y = n % 9 / 3 * 3;
 
    /* 判断n所在的小九宫格是否合法 */
    for (int i = x; i < x + 3; i++)
    {
        for (int j = y; j < y + 3; j++)
        {
            if (num[i][j] == key) return false;
        }
    }
 
    /* 全部合法，返回正确 */
    return true;
}

/* 深搜构造数独 */
int DFS(int n)
{
    /* 所有的都符合，退出递归 */
    if (n > 80)
    {
//    	cout<<"end"<<endl;
//		for(int i=0;i<9;i++){
//			for(int j=0;j<9;j++){
//				arr[i][j]=num[i][j];
//			}
//		}
		    Output();
//		system("pause");
        sign = true;
        return 0;
    }
    /* 当前位不为空时跳过 */
    if (num[n/9][n%9] != 0)
    {
        DFS(n+1);
    }
    else
    {
        /* 否则对当前位进行枚举测试 */
        for (int i = 1; i <= 9; i++)
        {
            /* 满足条件时填入数字 */
            if (Check(n, i) == true)
            {
//            	cout<<n<<" "<<i<<endl;
                num[n/9][n%9] = i;
                
                /* 继续搜索 */
                DFS(n+1);
                /* 返回时如果构造成功，则直接退出 */
//                if (sign == true) {
////                	cout<<"end"<<endl;
//                	return 0;
//				}
                /* 如果构造不成功，还原当前位 */
                num[n/9][n%9] = 0;
            }
        }
    }
}



//堡垒数独
// 3 0 2 0 4 0 6
//000000000
//000000000
//057123640
//070000020
//000968000
//000005000
//391456872
//000090000
//004730000
