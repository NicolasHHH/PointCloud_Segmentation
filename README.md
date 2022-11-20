# PointCloud_Segmentation

### Project Description

INF574 course Project : 
Implement from scratch the Model PointNet++ for point cloud segmentation. 

### Dataset Download link:
Credits to : https://github.com/AnTao97/PointCloudDatasets

- ShapeNetCore.v2 (0.98G)&ensp;[[TsinghuaCloud]](https://cloud.tsinghua.edu.cn/f/06a3c383dc474179b97d/)&ensp;[[BaiduDisk]](https://pan.baidu.com/s/154As2kzHZczMipuoZIc0kg)
- ShapeNetPart (338M)&ensp;[[TsinghuaCloud]](https://cloud.tsinghua.edu.cn/f/c25d94e163454196a26b/)&ensp;[[BaiduDisk]](https://pan.baidu.com/s/1yi4bMVBE2mV8NqVRtNLoqw)
- ModelNet40 (194M)&ensp;[[TsinghuaCloud]](https://cloud.tsinghua.edu.cn/f/b3d9fe3e2a514def8097/)&ensp;[[BaiduDisk]](https://pan.baidu.com/s/1NQZgN8tvHVqQntxefcdVAg)
- ModelNet10 (72.5M)&ensp;[[TsinghuaCloud]](https://cloud.tsinghua.edu.cn/f/5414376f6afd41ce9b6d/)&ensp;[[BaiduDisk]](https://pan.baidu.com/s/1tfnKQ_yg3SfIgyLSwQ2E0g)

### Github colabration workflow

```bash
# do once
git clone <repo in ssh>

# every time when opening the project
git pull # download updates from github to local 

# switch to personal repo
git checkout <branch name>

# check if you're in the right branch
git branch -l

>>>
  haiyang
  main
* tianyang

# commit 
git stauts # visualize modified and ignored files
git add <file names> # stage files for commiting 
git commit -m "commit message" # commit files in local
git push origin <branch name> # update github

# merge results on github using Pull request
# remember to add reviewer
Pull requests -> New Pull request
```

### Example
```shell
git pull 

git checkout tianyang

>>> 
Switched to branch 'tianyang'

### do some modifications in readme.md 
git status 

>>>
On branch tianyang
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   README.md
Untracked files:
  (use "git add <file>..." to include in what will be committed)
        .idea/

git add README.md 
git commit -m "updated readme"

git push origin tianyang 
```


