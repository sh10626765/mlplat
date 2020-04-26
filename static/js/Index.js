$(document).ready(function(){
	/*var web_width=document.body.clientWidth;
	var screen_width=window.screen.width;
	var scale=web_width/screen_width;
	document.getElementsByTagName('body')[0].style.zoom=0.9*scale;*/
	
	$("#index_ms").on("click",function(){
		
	});
	$(".index_app_pic").eq(0).on("click",function(){
		
	});
	$(".index_app").on("mouseover",function(){
		$(this).children('div').eq(1).css("background-color","#00CECE");
	});
	$(".index_app").on("mouseout",function(){
		$(this).children('div').eq(1).css("background-color","#008080");
	});
	/*for(var i = 0;i<$(".index_app").length;i++){
		$(".index_app").eq(i).on("mouseover",{index: i},function(event){
			var i = event.data.index;
			$(".index_app").eq(i).attr("color","red");
		})
	}*/
	/*for(var i = 0;i<rect.length;i++){
		rect.eq(i).on("mouseout",{index: i},function(event){
			var i = event.data.index;
			rect.eq(i).attr("stroke",col[i]);
			rect.eq(i).attr("stroke-width","1px");
		})
	}*/
});
