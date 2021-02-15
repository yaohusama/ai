//<pre code_snippet_id="1592833" snippet_file_name="blog_20160301_1_4048211" name="code" class="cpp">#include <iostream>
#include<iostream> 
#include<cstdio>
using namespace std;
int arr[9][9];
/* ������ɱ�־ */
bool sign = false;
//�������ڻҸ����ڣ��Ҹ�������������ֵĶ��а׸������� 
/* ������������ */
int num[9][9];
 
/* �������� */
void Input();
void Output();
bool Check(int n, int key);
int DFS(int n);
int siz;
int lix[100];
int liy[100];
/* ������ */
int main()
{
    cout << "������һ��9*9���������󣬿�λ��0��ʾ:" << endl;
    Input();
    DFS(0);


}
 
/* ������������ */
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
 
/* ����������� */
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
 
/* �ж�key����nʱ�Ƿ��������� */
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
    /* �ж�n���ں����Ƿ�Ϸ� */
    for (int i = 0; i < 9; i++)
    {
        /* jΪn������ */
        int j = n / 9;
        if (num[j][i] == key) return false;
    }
 
    /* �ж�n���������Ƿ�Ϸ� */
    for (int i = 0; i < 9; i++)
    {
        /* jΪn������ */
        int j = n % 9;
        if (num[i][j] == key) return false;
    }
 
    /* xΪn���ڵ�С�Ź����󶥵������� */
    int x = n / 9 / 3 * 3;
 
    /* yΪn���ڵ�С�Ź����󶥵������ */
    int y = n % 9 / 3 * 3;
 
    /* �ж�n���ڵ�С�Ź����Ƿ�Ϸ� */
    for (int i = x; i < x + 3; i++)
    {
        for (int j = y; j < y + 3; j++)
        {
            if (num[i][j] == key) return false;
        }
    }
 
    /* ȫ���Ϸ���������ȷ */
    return true;
}

/* ���ѹ������� */
int DFS(int n)
{
    /* ���еĶ����ϣ��˳��ݹ� */
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
    /* ��ǰλ��Ϊ��ʱ���� */
    if (num[n/9][n%9] != 0)
    {
        DFS(n+1);
    }
    else
    {
        /* ����Ե�ǰλ����ö�ٲ��� */
        for (int i = 1; i <= 9; i++)
        {
            /* ��������ʱ�������� */
            if (Check(n, i) == true)
            {
//            	cout<<n<<" "<<i<<endl;
                num[n/9][n%9] = i;
                
                /* �������� */
                DFS(n+1);
                /* ����ʱ�������ɹ�����ֱ���˳� */
//                if (sign == true) {
////                	cout<<"end"<<endl;
//                	return 0;
//				}
                /* ������첻�ɹ�����ԭ��ǰλ */
                num[n/9][n%9] = 0;
            }
        }
    }
}



//��������
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
