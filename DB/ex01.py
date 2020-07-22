''' 
select * from member
/* => 주석처리
from 이후에는 내가 생성한 db 테이블 이름 F5를 눌러서 실행하면
테이블에서 생성한 데이터를 보여줌
*/


--데이터 베이스 구축하기
--데이터 정의어(DDL) : 데이터베이스 만들기
create database Test02;

/*
create database <database명>
위의 쿼리문은 데이터 정의어(DDL) 중의 하나인 create문을 이용하는 쿼리입니다.

위의 쿼리문을 실행시키기 위해서 해당 쿼리문을 블록처리하고 F5를 눌러 실행시킵니다.

그리고 좌측의 개체탐색기 > 데이터베이스를 확인하면 Test02 라는 데이터베이스가 새로 생긴것을 확인할 수 있습니다.


이제 우리가 방금 생성한 Test02 라는 데이터베이스 내에 새로운 테이블을 생성하고 데이터를 추가해야 합니다.

하지만 우리가 처음 시작할 때 master 로 설정하고 시작한 것을 기억하시나요?

이 상태에서 테이블을 생성하거나 데이터를 입력하려고 하면 우리가 원하는대로, Test02 라는 데이터베이스에 데이터가 기록되지 않고 시스템 데이터베이스에 기록되게 됩니다.

따라서 우리가 앞으로 Test02에서 작업하겠다고 컴퓨터에게 알려주어야 합니다.

이를 위해서 아래와 같은 쿼리를 입력합니다.

use Test02;

위의 쿼리문을 실행하면 아래와 같이 master로 선택되어 있었던 것이 Test02로 바뀜

'''

'''


create table member(
	id int constraint pk_code primary key,
	name char(10),
	email char(10)
);

/*
쿼리를 실행시킬 때는 실행시키고자 하는 부분만 블록으로 감싸 F5를 눌러야한다.
그렇지 않고 F5를 누르게되면 해당 쿼리창의 시작부터 끝까지 모든 쿼리가 다시 실행되므로 에러가 발생할 수 있다.
id 칼럼은 contraint pk_code primary key 라고 붙어있는데, 여기서 constraint는 해당 칼럼에 특정 제약조건을 주겠다라는 의미이고 그 제약조건의 내용이 뒤에 따라서 붙습니다
여기서 pk_code primary key 라는 제약조건이 붙었는데, 이는 pk_code 라는 이름의 primary key로 설정하겠다라는 의미입니다.
즉, member 테이블에서의 primary key, 기본키는 id컬럼이며 해당 기본키의 이름은 pk_code이다
*/

-- 데이터 조작어(DML) : INSERT, SELECT

insert into member values(10, '홍범우', 'hong@eamil');
/*
위의 쿼리는, member 라는 테이블에 데이터를 insert 할 것이다라는 의미
입력되는 데이터의 내용은 values(~~~) 내부에 입력
그리고 입력한 데이터가 잘 저장되었나 확인하기 위해 아래 쿼리를 입력
select * from member; 이게 확인하기 위한 쿼리
* : *는 모든 칼럼을 의미 배경이되는 테이블은 from ~~

*/

select * from member
'''
