// pages/query_interface/query_interface.js
var util=require('../../utils/util.js');
var app=getApp();
Page({

  /**
   * 页面的初始数据
   */
  data: {
    inputPos:'',
    time:'',
    brpos:[[1,0]],
    chooseId:"90",
    chooseDir:0,
    chooseBool:0,
    appointment:'',
    full:0,
    queue_num:0,
    st_wait:0,
    end_wait:0,
    profile:{},
  },
  
  //事件处理函数
  stillWhite: function(e) {
    wx.showToast({
      title:'选择成功',
      icon:'success',
      // image:'',
      duration:2000
    });
    console.log(e.target.id);
    var idtmp=e.target.id;
    var id_tmp=idtmp.slice(1,idtmp.length);
    var room=0;
    if(id_tmp=='right'){
      room=1;
    }
    // console.log(idtmp);
    var time_tmp=util.formatTime(util.calTime());
    this.setData({
      chooseId:idtmp[0],
      chooseDir:room,
      chooseBool:1,
      appointment:time_tmp,
    })

  },
  personal_func: function() {
    wx.navigateTo({
      url: '../personal_interface/personal_interface',
    })
  },
  purpleAlready: function() {
    wx.showToast({
      title:'已有人',
      icon:'loading',
      // image:'',
      duration:2000
    });
  },
  chooseTap:function() {
    if(this.data.chooseBool==1){
      // wx.setStorageSync({
      //   data: this.data,
      //   key: '1',
      // });
      try {
        wx.setStorageSync('1', this.data)
      }catch(e) {console.log(e)}
    wx.navigateTo({
      url: '../success_interface/success_interface'
    });

  }
  else {
    wx.showToast({
      title:'请先选择',
      icon:'loading',
      duration:2000,
    })
  }
  },
  wait_count:function(){
    try {
      wx.setStorageSync('2', this.data)
    }catch(e) {console.log(e)};
    wx.navigateTo({
      url: '../count_down_interface/count_down_interface',
    });
  },
  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    var profile=app.globalData.userInfo;
    wx.getStorageInfo({
      success: (res) => {
      if (res.keys.includes('0') ){
        var value = wx.getStorageSync("0");
        this.setData({
          inputPos:value.inputPos,
          profile:profile,
        });
      } 
    
  }

  })
  var time=util.formatTime(new Date());
  this.setData({time:time});
  var br_pos=util.getData();
  // console.log(br_pos.data.people_at);
  this.setData({
    brpos:br_pos.data.people_at,
    full:br_pos.data.full,
    queue_num:br_pos.data.queue_num,
    st_wait:br_pos.data.st_wait,
    end_wait:br_pos.data.end_wait,

  })
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})