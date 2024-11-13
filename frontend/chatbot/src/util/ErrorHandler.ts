import Swal from "sweetalert2";

export const ErrorHandler = {
  showError: (error: Error) => {
    console.error('Error occurred:', error);
    
    Swal.fire({
      text: error.message || '오류가 발생했습니다.',
      icon: 'error',
      confirmButtonText: '확인',
      customClass: {
        confirmButton:
          'bg-blue-500 text-white px-4 py-2 rounded-lg focus:ring-0 focus:outline-none active:bg-blue-500 hover:bg-blue-600',
      },
      buttonsStyling: false,
    });
  }
 };
 