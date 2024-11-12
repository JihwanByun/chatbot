import { Meta, StoryObj } from '@storybook/react';
import CardWithModal, { CardWithModalProps } from './CardWithModal';

const meta: Meta<typeof CardWithModal> = {
  title: 'COMPONENTS/ATOMS/CARD/CardWithModal',
  component: CardWithModal,
  tags: ['autodocs'],  // 자동 문서화를 위해 사용
  argTypes: {
    title: { control: 'text' },
    content: { control: 'text' },
  },
};

export default meta;

type Story = StoryObj<CardWithModalProps>;

export const Default: Story = {
  args: {
    title: '출장 규정 (제 1장 / 1조)',
    content: '제 1항 규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정규정',
  },
};
