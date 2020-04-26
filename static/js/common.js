$(document).ready(function(e) {
	var selectedElements=$(".text_1").val();
	for(var i=0;i<$(".text_3 a").length;i++){
		$(".text_3 a").eq(i).bind("click", {index: i}, function(event){
			var i=event.data.index;
			$(".search_method").html($(".text_3 a").eq(i).html());
		});
	}
	
	for(var i=0;i<$(".element").length;i++){
		$(".element").eq(i).bind("mousedown", {index: i}, function(event){
			var i=event.data.index;
			$(".element").eq(i).css("border","#CCC 2px inset");
		});
		$(".element").eq(i).bind("mouseup", {index: i}, function(event){
			var i=event.data.index;
			$(".element").eq(i).css("border","#CCC 2px outset");
		});
		$(".element").eq(i).bind("click", {index: i}, function(event){
			var i=event.data.index;
			selectedElements = $(".text_1").val() + $(".element").eq(i).prop("lastChild").nodeValue + '-';
			$(".text_1").val(selectedElements);
		});
	}
	
	$(".clear").on("click",function(){
		$(".text_1").val("");
	})
	
	function toSubscript(str){
		return str.replace(/\d/g,function(val){
			return (val+"").sub();
		})
	}
	
	function toDecimal2(x){
		var f = parseFloat(x);
		if(isNaN(f)){
			return false;
		}
		var f = Math.round(x*100)/100;
		var s = f.toString();
		var rs = s.indexOf('.');
		if(rs<0){
			rs = s.length;
			s += '.';
		}
		while(s.length <= rs + 2){
			s += '0';
		}
		return s;
	}
	function toDecimal3(x){
		var f = parseFloat(x);
		if(isNaN(f)){
			return false;
		}
		var f = Math.round(x*1000)/1000;
		var s = f.toString();
		var rs = s.indexOf('.');
		if(rs<0){
			rs = s.length;
			s += '.';
		}
		while(s.length <= rs + 3){
			s += '0';
		}
		return s;
	}
	
	////////////////////////////////button_pic//////////////////
	var btm_pic = '<img src="static/img/图标合辑.png" class="bottom_logo" style="">';
					/*'<div class="btm_left" style="width:35%;height:100%;float:left">' + 
						'<img class="btnPic" src="static/img/学校图标/北京科技大学.png" width="160" height="40" style="float:left;margin:5px auto auto 40px">' + 
						'<img class="btnPic" src="static/img/学校图标/华南理工大学.png" width="160" height="40" style="float:left;margin:5px auto auto 40px">' + 
						'<img class="btnPic" src="static/img/学校图标/南方科技大学.png" width="140" height="45" style="float:left;margin:5px auto auto 40px">' +  
						'<img class="btnPic" src="static/img/学校图标/四川大学.png" width="130" height="40" style="float:left;margin:5px auto auto 40px">' +  
					'</div>' + 
			'<img src="static/img/bottom_logo.png" class="bottom_logo" style="margin-top:2%">' +  
					'<div class="btm_left" style="width:35%;height:100%;float:right;">' + 
						'<img class="btnPic" src="static/img/学校图标/广东省材料与加工研究所.png" width="70" height="65" style="float:right;margin:5px 20px auto auto">' + 
						'<img class="btnPic" src="static/img/学校图标/清华大学深圳研究生院.png" width="120" height="75" style="float:right;margin:5px 20px auto auto">' + 
						'<img class="btnPic" src="static/img/学校图标/上海交通大学.png" width="130" height="40" style="float:right;margin:5px 20px auto auto">' + 
						'<img class="btnPic" src="static/img/学校图标/中山大学.png" width="130" height="35" style="float:right;margin:5px 20px auto auto">' + 
					'</div>'*/
					
	$(".bottom_").html(btm_pic);
	$(".bottom_").css("padding-top",0);
	
});