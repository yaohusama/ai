//<pre code_snippet_id="1592833" snippet_file_name="blog_20160301_1_4048211" name="code" class="cpp">#include <iostream>
#include<iostream> 
#include<cstdio>
using namespace std;
int arr[9][9];
/* ������ɱ�־ */
bool sign = false;
 
/* ������������ */
int num[9][9];
 
/* �������� */
void Input();
void Output();
bool Check(int n, int key);
int DFS(int n);
int siz;int block;
int lix[100][100];
int liy[100][100];
/* ������ */
int main()
{
    cout << "������һ��9*9���������󣬿�λ��0��ʾ:" << endl;
    Input();
    DFS(0);

//Output();
}
 
/* ������������ */
void Input()
{
	cin>>siz>>block;
	for(int j=0;j<block;j++)
	for(int i=0;i<siz;i++){
		cin>>lix[j][i]>>liy[j][i];
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
            /* ��������ʱ�������� */bool flag=0;
            if (Check(n, i) == true)
            {
//            	cout<<n<<" "<<i<<endl;
                num[n/9][n%9] = i;
                int x=n/9; int y=n%9;int ttmp=-1;
            	for(int j=0;j<siz;j++){
            		if(x==lix[0][j]&&y==liy[0][j]){
            			ttmp=j;
            			for(int k=1;k<block;k++){
            				int xx=lix[k][j];int yy=liy[k][j];int nn=xx*9+yy;
            				if(!Check(nn,i)){
            					flag=true;
            					break;
							}
							else{
								num[xx][yy]=i;
							}
						}
						break;
					}


				}
				if(flag) 
				{
					for(int k=0;k<block;k++){
            		int xx=lix[k][ttmp];int yy=liy[k][ttmp];//int nn=xx*9+yy;
            			num[xx][yy]=0;
				}
				continue;
			}
                /* �������� */
                DFS(n+1);
                /* ����ʱ�������ɹ�����ֱ���˳� */
                if (sign == true) {
//                	cout<<"end"<<endl;
                	return 0;
				}

                /* ������첻�ɹ�����ԭ��ǰλ */
                num[n/9][n%9] = 0;
            }
        }
    }
}
//�������� 

//4 4
//0 2 0 3 1 2 1 3
//2 7 2 8 3 7 3 8
//5 0 5 1 6 0 6 1
//7 5 7 6 8 5 8 6
//040000000
//010000425
//058300100
//000000700
//000000000
//005000000
//001006250
//592000040
//000000090
