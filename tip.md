1)LDA
2)分解 tag, find in desc
3)辅助特征, 当前相似的词在文中出现的次数
4)tf-idf (基于type_name)
5)include(total, partial)
6)app type_name,分大组
7)从desc 过滤相似词
8)n-gram 去除?
9)手工tocken
10)解释性
11)倒序挑选字符串

99998 app_desc.dat
29999 apptype_train.dat

1134  records have multiply type_id

130097 total


异常app:
1A23C73F4F3E892E2A9DF3C338B80313   -102
D675211C835694A9F096B3AD3C8A9F79   -102
59913551A1752422F3B191E0C353309D   -102
6FFCF1564CFA7547DEEEB5DDCC83A24B   -102
E2CC4670C695BFD41FC4ABFDE95C7B36   -102
0C8C840A534F32D8A608F50D67663E83   -102
D396674F43367C4FDEF82CDA78756D4F   -102
1F0AF6FA7424660692173FF4134903CB   -102

#Manual handle
egrep dkplugin  input/0823/*


no case for 140208

grep 'com.android.iconnect' ./input/0823/*

唐小僧

WXEntryActivity

辽宁和教育

leak between test and train

https://android.myapp.com/myapp/detail.htm?apkName=com.kiees.android
vs
https://android.myapp.com/myapp/detail.htm?apkName=com.kiess


grep 百度手机助手 ./input/0823/*

name first place

name partially match

融合,交叉数据源, 去除空格,大小写 一半a, 一半b



 #ffababab       com.hisense.uifac
 FACTORY MENU    com.hisense.uifac
 工厂菜单        com.hisense.uifac


aiqianjin.jiea

WXEntryActivity

开心躲猫猫
当妈模拟器
