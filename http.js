const http = require('http');
const fs = require('fs');

http.createServer((req, res) => {
    const fileName = '.' + req.url;
    console.log(fileName);
    fs.readFile(fileName, 'utf8', (err, data) => {
        if(err){
            console.log('文件读取失败');
            res.write('404');
        }else{
            console.log('文件读取成功');
            res.writeHead(200, {'Content-Type': 'text/html;charset=utf-8'});
            res.write(data);
        }
        res.end();
    });
}).listen(3001, '127.0.0.1', () => {
    console.log('服务器已开启');
})

// 参考：https://blog.csdn.net/maidu_xbd/article/details/86547146