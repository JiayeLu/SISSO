! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.


module var_global
!=======================================================================
! 全局变量与参数定义模块
! - 供 SISSO 主程序、FC/DI 等模块共享
! - 含运行时间统计、LASSO 参数、数据/特征全局配置等
!=======================================================================
use mpi

implicit none

! 运行时间统计结构体
type run_time ! parameters to show the program execution time
  real*8 sFCDI,eFCDI,sFC,eFC,sDI,eDI
end type

! LASSO 超参数与控制开关
type LASSOpara  ! parameters for LASSO
  integer max_iter,nlambda,dens,nl1l0
  real*8  tole,minrmse,elastic
  logical warm_start,weighted
end type LASSOpara

type(run_time) mytime   ! 运行时间记录
type(LASSOpara) L1para  ! LASSO 参数集合

! ------------------- 核心控制参数 -------------------
integer   nsf,ntask,nunit,fcomplexity,rung,str_len,nf_DI,nf_L0,ptype,Smaxlen,&
          desc_dim,nmodel,iFCDI,fileunit,task_weighting,npoint,restart,fstore
real*8    fmax_min,fmax_max,bwidth,PI
parameter (str_len=150,PI=3.14159265d0,Smaxlen=60)
character ops(20)*200,method_so*10,metric*10   ! 操作符集合、稀疏求解方法、评价指标
integer*8 nf_sis(10000),nf_sis_avai(10000)     ! SIS 目标数量与可用数量
logical   fit_intercept,scmt                   ! 是否拟合截距、是否约束单调性
integer,allocatable:: nsample(:),ngroup(:,:),isconvex(:,:) ! 样本数/分组/凸域标记
real*8,allocatable:: target_y(:),pfdata(:,:),res(:),feature_units(:,:),ypred(:)
character(len=30),allocatable:: pfname(:)      ! 原始特征名称
integer   mpierr,mpirank,mpisize,status(MPI_STATUS_SIZE) ! MPI 状态与信息

! ------------------- 描述符级特征约束 -------------------
integer, parameter :: max_allowed_features=10000
logical, allocatable :: has_restriction(:)  ! 是否启用特征限制
integer, allocatable :: allowed_features(:,:), allowed_features_n(:) ! 允许特征索引

end module 
