console.log('index-同步任务');
Promise.resolve().then((res) => {
  console.log('index-Promise');
});
setTimeout(() => {
  console.log('index-setTimeout');
}, 1000);