// const reg1 = /setdl\/(xinwen|huodong|gonglue)*/;
// const reg2 = /setdl\/wap/;

// let str1 = 'http://www.37.com/setdl/xx/123';
// let str2 = 'http://www.37.com/setdl/wap/';
// console.log(str1.match(reg1));
// console.log(str2.match(reg2));

server
{
	listen       80;
	server_name  37.com.cn www.37.com.cn;
        #rewrite ^/.*/([a-z0-9]*)/ http://37.com.cn/$1/ break;
        #rewrite ^/(.*)/(.*)/ http://37.com.cn/$1/$2/ break;
        if ($host != '37.com.cn' ) {
		rewrite ^/(.*)$  http://37.com.cn/$1   permanent;
	}
	index index.html index.htm index.php index.shtml;
	root  /www/m.37.com;
rewrite ^/wap/(.*)$ http://37.com.cn/m/ permanent;
rewrite ^/dts/wap/(.*)$ http://37.com.cn/yhjy/wap/ permanent;
rewrite ^/dts/(.*)$ http://37.com.cn/yhjy/ permanent;
rewrite ^/spread/dhxy/baidu/(.*)$ http://37.com.cn/dhxy/  permanent;

rewrite ^/(snxyj|jstm|zs|jzgmd|yxzg|sdsxs|shqz|dntgol|hxzhg|yxws|blr2|wzry|fsyxb|zzhxjx|jl|hjzg|sgcq|zshh|fjlrs|wdsj|wdkl|lyzdy|gcd3D|brdz|zjcq|wd|czsg|qmxx2|zshh|xyjh|yjqy|dfzj|zjzr|tt2xm|qyz|frxxz|tx|yys|yqzjxy|jyzj3D|zyzn|mfyx|cyhx|jczyt|dtws|ttry|zmq|jysgqy|gmdl|bzfs|mlsj|dfbb|rzds|tjxmll|dtry|xycq|cyzr|wjdld|hyxd|wszzlgf|ly|bdtx|mf|xyjzsdbgj|ahzh|mljx|qnyh|shjzcycs|tz|mxzg|sjh2|jyzj|bhxy|qqhcs|hszz|gfxm|qrz|snsgz|qmgj|ldxy|ssjx|qh97ol|xhqg|dmol|jcyt|xysj|snmst|klsd|glxbwysw|dzz|gfsl|gmflb|dhhsd5|zyzj|gyc|qmxx|zt|qmqj|ahlm|ahzs|xxjqxz|sg|rxgx|jjdjr2|tjb|hhwqzzl|xxqy|tjxm2|cftg|txxtx|wyft|sdyxz|thdmx|rxxj|fmz|fmz|sj2|zhsdj|mtj|rxjb|csbyx|xyxmp|qmgj|bm|qmly|dmyx|hycs|zhslm|xysmj|mjzr|dtslz|lyjl|wdhl|mlsd|tjxm|txhd|tlbb|qjxy|hdzb|tfqst|wsjj|qy2|xxqy|yd|lyzr|shyx|hagn|xmhzc|sq|sanguo|sm|ahnw|jl|wzzj|kd|gdjh|tfqst|dtcq|zmyx|hhwqh|mmasg|jy|bsbzg|qsmy|jlb|dmbj|gwlr|dpsg|zscq|qmsjb|mgld|bzwx|lsq|liubei|ttaxy|mjh|sxd|aqgy|wltx|astd|cjdt|qmqz|ldxy|lqjx|xsd|ysqth|dntg|mhsy|sqsd|bjx|lt|fbhx|sfyyq|bxjgqx|lmcs|xqsmy|shjx|mxywk|mds|mxxc|jshm|zlzr|jdzj|mlys|dgtf3|lmbj|dmx|glkp|byhonline|lyb|skzr|mlxm|yzg|zezc|NBAftx|lycx|sqszg|cssg|xmdmlrj|slZero|jlgj|cosyxt|dnsg|yxws|rxtk|snsdmb|nnj|swgdmw)/(.*)$ http://37.com.cn/404.html permanent;

	include gzip.conf;
	rewrite ^/invite.html$ http://37.com.cn redirect;
	rewrite ^/zsgkd/(.*)$  http://37.com.cn/kd/ permanent;
        #rewrite ^/.*/([a-z0-9]*)/ http://37.com.cn/$1/ permanent;
        rewrite ^/shop/(.*)$ http://37.com.cn/404.html permanent;

if ($request_uri ~ "/sdk-wrap/service/(\?*)(.*)" ) {
    return 444;
}
if ($request_uri ~ "/sdk-wrapV2/service/(\?*)(.*)" ) {
    return 444;
}

# 灰度
location ~ /v2/payment($|/) {
  try_files $uri $uri/ /v2/payment/index.html;
}

location /service-system/ {
try_files $uri $uri/ /service-system/index.html;
}
location /mt/ {
try_files $uri $uri/ /mt/index.html;
}
location /payment/ {
try_files $uri $uri/ /payment/index.html;
}
location /syqd-admin/ {
try_files $uri $uri/ /syqd-admin/index.html;
include whiteip.txt;
}
location /gamefaq/ {
try_files $uri $uri/ /gamefaq/index.html;
}
location /jlsd/ {
rewrite ^/jlsd/(.*)  /mus/$1 permanent;
}
location /mus/ {
try_files $uri $uri/ /mus/index.html;
}
location /mwysc/ {
try_files $uri $uri/ /mwysc/index.html;
}
location /community/ {
try_files $uri $uri/ /community/index.html;
}
#location /qhjx/ {
#try_files $uri $uri/ /qhjx/index.html;
#}
location /m/ {
try_files $uri $uri/ /m/index.html;
}
location /hdqyx/ {
try_files $uri $uri/ /hdqyx/index.html;
}
location /user-system/ {
try_files $uri $uri/ /user-system/index.html;
}
location /mrsc/ {
try_files $uri $uri/ /mrsc/index.html;
}
location /vip/ {
try_files $uri $uri/ /vip/index.html;
}
location /ysczg/ {
try_files $uri $uri/ /ysczg/index.html;
}
#location /wpgx/ {
#try_files $uri $uri/ /wpgx/index.html;
#}
location /fsyhj/ {  
try_files $uri $uri/ /fsyhj/index.html;
}
location /platform/ {
  try_files $uri $uri/ /platform/index.html; 
}
location /gift-center/ {
    try_files $uri $uri/ /gift-center/index.html; 
}
location /yqklnw/ {
    try_files $uri $uri/ /yqklnw/index.html; 
}
location /pxr/ {   
   try_files $uri $uri/ /pxr/index.html; 
}
location /wlzh/ {   
   try_files $uri $uri/ /wlzh/index.html; 
}
location /brzhg/ {   
   try_files $uri $uri/ /brzhg/index.html; 
}
location ~ /whjx($|/) {
   try_files $uri $uri/ /whjx/index.html;
}
location ~ /mjzz($|/) {
   try_files $uri $uri/ /mjzz/index.html;
}
location ~ /mtdl($|/) {
   try_files $uri $uri/ /mtdl/index.html;
}
location ~ /jsxw($|/) {
   try_files $uri $uri/ /jsxw/index.html;
}
location ~ /brzhg($|/) {
   try_files $uri $uri/ /brzhg/index.html;
}
location ~ /ydwx($|/) {
   try_files $uri $uri/ /ydwx/index.html;
}
location ~ /hsdj($|/) {
   try_files $uri $uri/ /hsdj/index.html;
}
#临时跳转404,上面注释location wpgx/qhjx
location  /wpgx/ {
   return 404;
}
location  /qhjx/ {
   return 404;
}
location  /hdqy/ {
   return 404;
}


#临时跳转到维护页
location ~ /platform($|/) {
   try_files $uri $uri/ /platform/index.html;
}
location ~ ^/gift($|/) {
   rewrite  ^/(.*) /platform/gift/ redirect;
}
location ~ ^/paycenter/index.html($|/) {
   rewrite  ^/(.*) /platform/paycenter/ redirect;
}
location ~ ^/page/9406.html($|/) {
   rewrite  ^/(.*) /platform/about/ redirect;
}
location ~ ^/service($|/) {
   rewrite  ^/(.*) /platform/service/ redirect;
}
location ~ ^/usercenter/accountinfo($|/) {
   rewrite  ^/(.*) /platform/usercenter/ redirect;
}
location ~ ^/gamecenter($|/) {
   rewrite  ^/(.*) /platform/gamecenter/ redirect;
}
location /awlzw/ {
    return 404;
}
location = /login.html {
   rewrite  ^/(.*) https://$host/platform/login/ redirect;
}
location ~ ^/login($|/) {
   rewrite  ^/(.*) https://$host/platform/login/ redirect;
}
location ~ ^/register($|/) {
   rewrite  ^/(.*) https://$host/platform/register/ redirect;
}
location = /register.html {
   rewrite  ^/(.*) https://$host/platform/register/ redirect;
}
location = / {
   rewrite  ^/(.*) /platform/ redirect;
}


add_header Access-Control-Allow-Origin http://api.37wandtsh5.5jli.com;
add_header Access-Control-Allow-Headers X-Requested-With,Content-Type;
add_header Access-Control-Allow-Methods GET,POST,OPTIONS;

	location ~ .*\.(php|php5)$
	{      
#fastcgi_pass  unix:/tmp/php-cgi.sock;
		fastcgi_pass  127.0.0.1:9000;
		fastcgi_index index.php;
		include fcgi.conf;
		access_log  logs/php.access.log main;
	}
	error_page 404  /404.html;
	error_page 403  /404.html;
	error_page 500  /500.html;
	location ~ .*\.(gif|jpg|jpeg|png|bmp|swf)$
	{
		expires      30d;
	}
	location ~ .*\.(js|css)?$
	{
		expires      1h;
	}    
	access_log  logs/access.log  main;
        location  ~ 404.html 
        {
          access_log  logs/php.access.log  main;
        } 
include safe_rule.conf;
#access_by_lua_file /usr/local/webserver/openresty/nginx/access_limit.lua;
}
