export class AuthServiceError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public field?: string
  ) {
    super(message);
    this.name = 'AuthServiceError';
  }
}