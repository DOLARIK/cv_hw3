# Only for development purposes

1. How to build an image from dockerfile:
```bash
docker build -f Dockerfile.dev -t cv_hw_3.dev .
```

2. For running development container:
```bash
docker run --name cv_hw_3.dev -p 13888:8888 -v ${pwd}:/assignment_3/ -itd cv_hw_3.dev
```

Now, visit the url: [localhost:13888](http://localhost:13888) (password: `dev`)