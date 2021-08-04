const http = require('http');
const querystring = require('querystring');

const server = http.createServer();

server.on('request', (req, res) => {
    const params = querystring.parse(req.url.split('?')[1]);
    const fn = params.callback;
    res.writeHead(200, { 'Content-Type': 'text/javascript' });
    res.write(fn + '(' + JSON.stringify(params) + ')');

    res.end();
})

server.listen('8080', '127.0.0.1', () => {
    console.log('Server is running at port 8080...');
});