const formatTime = date => {
  const year = date.getFullYear()
  const month = date.getMonth() + 1
  const day = date.getDate()
  const hour = date.getHours()
  const minute = date.getMinutes()
  const second = date.getSeconds()

  // return `${[year, month, day].map(formatNumber).join('/')} ${[hour, minute, second].map(formatNumber).join(':')}`
  return `${[hour, minute].map(formatNumber).join(':')}`
}

const formatNumber = n => {
  n = n.toString()
  return n[1] ? n : `0${n}`
}

module.exports = {
  formatTime
}
var br8_3=require('../data/br8_3');
function getData(){
  return br8_3.br8_3;
}
function calTime(){
  var date=new Date();
  // console.log(date)
  var res=date.setMinutes(date.getMinutes()+2);
  var date_tmp=new Date(res);
  // console.log(date_tmp)
  return date_tmp;
}
module.exports.getData = getData;
module.exports.calTime = calTime;