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
//040517060
//200040009
//003000500
//500401007
//920000085
//700905001
//008000700
//600070008
//050869030

//�������� 

//090507010
//306000902
//010000070
//700010004
//000302000
//200050007
//020000060
//605000709
//080605040

//��¡���� 
//040000000
//010000425
//058300100
//000000700
//000000000
//005000000
//001006250
//592000040
//000000090


//��ʽ����
//059000610
//800060005
//600309008
//006030400
//090406080
//001090700
//100507003
//900010007
//065000820 

//��������
//002905700
//090201050
//604000201
//930000082
//000000000
//420000079
//809000506
//040806030
//003509800
