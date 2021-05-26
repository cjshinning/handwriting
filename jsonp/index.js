const http = require('http');
const urllib = require('url');

let data = {'data': 'world'};

http.createServer((req, res) => {
    let params = urllib.parse(req.url, true);
    if(params.query.callback){
        console.log(params.query.callback)
        const str = params.query.callback + '(' + JSON.stringify(data) + ')';
        res.end();
    }else{
        res.end();
    }
}).listen(8080, ()=>{
    console.log('jsonp server is on');
})