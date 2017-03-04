注意：
1. 实时计算有三种业务场景，对应三个文件夹，分别对应用户收藏一本图书，取消收藏一本图书及新用户首次注册实时推荐。
2. 关于代码所依赖的包还在整理中。完善后统一上传
3. 目前除新用户首次注册代码未运行外，其余两个模块均在服务器Master虚拟机上测试运行
4. linux 后台分别运行三个进行即可
5. nohup python guessLike_u_first_select.py > python.log >&1 &
6. nohup python guessLike_u_like.py > python.log >&1 &
7. nohup python guessLike_u_nlike.py > python.log >&1 &
