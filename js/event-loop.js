// process.nextTick(() => {
//     console.log(1); 
// });
// new Promise((resolve) => {
//     resolve();
// }).then(() => {
//     console.log(2);
// });
// process.nextTick(() => {
//     console.log(3); 
// });

// process.nextTick 永远大于 promise.then

function test(){
    console.log('start');
    setTimeout(() => {
        console.log('children2');
        Promise.resolve().then(() => {
            console.log('children2-1');
        })
    }, 0)
    setTimeout(() => {
        console.log('children3');
        Promise.resolve().then(() => {
            console.log('children3-1');
        })
    }, 0)
    Promise.resolve().then(() => {
        console.log('children1');
    })
    console.log('end');
}

test();

// node 11及以上的版本，跟浏览器一致（每执行一个宏任务就执行完微任务队列）
// start
// end
// children1
// children2
// children2-1
// children3
// children3-1

// node 11以下版本（先执行所有宏任务的结果，再执行微任务）
// start
// end
// children1
// children2
// children3
// children2-1
// children3-1