module bench_utils
  use iso_c_binding, only: c_int, c_long, c_double
  implicit none

  private
  public :: parse_i64_arg, parse_i32_arg, print_result, almost_equal_f32, wall_time

  integer(c_int), parameter :: fnacc_clock_monotonic = 1_c_int

  type, bind(C) :: c_timespec
     integer(c_long) :: tv_sec
     integer(c_long) :: tv_nsec
  end type c_timespec

  interface
     function c_clock_gettime(clk_id, tp) bind(C, name="clock_gettime") result(ierr)
       import :: c_int, c_timespec
       integer(c_int), value :: clk_id
       type(c_timespec) :: tp
       integer(c_int) :: ierr
     end function c_clock_gettime
  end interface

contains

  subroutine parse_i64_arg(index, default_value, value)
    integer, intent(in) :: index
    integer(8), intent(in) :: default_value
    integer(8), intent(out) :: value

    character(len=128) :: arg
    integer :: status

    value = default_value

    call get_command_argument(index, arg, status=status)
    if (status == 0) then
       read(arg, *, iostat=status) value
       if (status /= 0) value = default_value
    end if
  end subroutine parse_i64_arg

  subroutine parse_i32_arg(index, default_value, value)
    integer, intent(in) :: index
    integer, intent(in) :: default_value
    integer, intent(out) :: value

    integer(8) :: tmp

    call parse_i64_arg(index, int(default_value, 8), tmp)
    value = int(tmp)
  end subroutine parse_i32_arg

  subroutine print_result(name, n, m, reps, seconds, bytes_per_rep, errors)
    character(len=*), intent(in) :: name
    integer(8), intent(in) :: n
    integer(8), intent(in) :: m
    integer, intent(in) :: reps
    real(8), intent(in) :: seconds
    real(8), intent(in) :: bytes_per_rep
    integer, intent(in) :: errors

    real(8) :: avg
    real(8) :: bandwidth

    avg = seconds / real(reps, 8)
    bandwidth = bytes_per_rep / avg / 1.0d9

    write(*,'(a,",",i0,",",i0,",",i0,",",es16.8,",",es16.8,",",i0)') &
         trim(name), n, m, reps, avg, bandwidth, errors
  end subroutine print_result

    logical function almost_equal_f32(got, expected) result(ok)
    real, intent(in) :: got
    real, intent(in) :: expected

    real :: diff
    real :: scale
    real :: tol

    diff = abs(got - expected)
    scale = max(1.0, abs(got), abs(expected))

    ! Single precision benchmark tolerance.
    !
    ! This is intentionally relative because different GPU/CPU backends may use
    ! fused multiply-add or different reassociation/rounding.
    tol = max(1.0e-3, 1.0e-4 * scale)

    ok = diff <= tol
  end function almost_equal_f32

  real(c_double) function wall_time()
    type(c_timespec) :: ts
    integer(c_int) :: ierr
    
    ierr = c_clock_gettime(fnacc_clock_monotonic, ts)

    if (ierr /= 0_c_int) then
      wall_time = 0.0_c_double
    else
      wall_time = real(ts%tv_sec, c_double) + &
                  1.0e-9_c_double * real(ts%tv_nsec, c_double)
    end if
  end function wall_time

end module bench_utils

