module bench_utils
  implicit none

contains

  function wall_time() result(t)
    real(8) :: t
    integer :: count, rate

    call system_clock(count, rate)
    t = real(count, 8) / real(rate, 8)
  end function

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
  end subroutine

  subroutine parse_i32_arg(index, default_value, value)
    integer, intent(in) :: index
    integer, intent(in) :: default_value
    integer, intent(out) :: value

    integer(8) :: tmp

    call parse_i64_arg(index, int(default_value, 8), tmp)
    value = int(tmp)
  end subroutine

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
  end subroutine

  logical function almost_equal_f32(got, expected) result(ok)
    real, intent(in) :: got
    real, intent(in) :: expected

    real :: diff
    real :: scale
    real :: tol

    diff = abs(got - expected)
    scale = max(1.0, abs(expected))

    ! Single precision benchmark tolerance.
    tol = max(1.0e-3, 1.0e-5 * scale)

    ok = diff <= tol
  end function

end module

